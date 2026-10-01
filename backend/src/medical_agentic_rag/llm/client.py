"""Client google-genai trỏ tới endpoint shopaikey — một hàm run_task() duy nhất.

Node trong graph không gọi SDK trực tiếp: run_task() lo chọn model theo task,
output có schema, retry, trừ ngân sách và ghi log.
"""
from __future__ import annotations

import time
from typing import Optional, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from medical_agentic_rag import budget
from medical_agentic_rag.config import settings
from medical_agentic_rag.llm.tasks import TASKS
from medical_agentic_rag.observability import trace

T = TypeVar("T", bound=BaseModel)

_client: genai.Client | None = None

MODEL_BY_TIER = {
    "fast": settings.LLM_MODEL_FAST,
    "strong": settings.LLM_MODEL_STRONG,
    "judge": settings.LLM_MODEL_JUDGE,
}


class LLMTaskFailed(RuntimeError):
    pass


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=settings.SHOPAIKEY_API_KEY,
            http_options=types.HttpOptions(
                base_url=settings.GENAI_BASE_URL,
                timeout=settings.LLM_TIMEOUT_MS,
            ),
        )
    return _client


async def run_task(task: str, contents: str, state: dict, schema: Optional[type[T]] = None) -> T | str:
    spec = TASKS[task]
    client = get_client()
    cfg = types.GenerateContentConfig(
        temperature=spec.temperature,
        max_output_tokens=spec.max_output_tokens,
        response_mime_type="application/json" if schema else None,
        response_schema=schema,
    )
    last_exc: Exception | None = None
    for attempt in range(2):  # tối đa 1 lần thử lại
        budget.charge_llm(state, task)
        t0 = time.perf_counter()
        try:
            resp = await client.aio.models.generate_content(
                model=MODEL_BY_TIER[spec.model_tier], contents=contents, config=cfg,
            )
            out = schema.model_validate_json(resp.text) if schema else (resp.text or "")
            trace.log_llm(state, task, attempt, t0, ok=True)
            return out
        except (ValidationError, errors.APIError) as exc:
            last_exc = exc
            trace.log_llm(state, task, attempt, t0, ok=False, error=type(exc).__name__)
    raise LLMTaskFailed(f"{task}: {last_exc}")
