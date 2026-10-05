"""Client google-genai trỏ tới endpoint shopaikey — một hàm run_task() duy nhất.

Node trong graph không gọi SDK trực tiếp: run_task() lo chọn model theo task,
output có schema, retry, trừ ngân sách và ghi log.
"""
from __future__ import annotations

import re
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from medical_agentic_rag import budget
from medical_agentic_rag.config import settings
from medical_agentic_rag.errors import LLMTaskFailed
from medical_agentic_rag.llm.tasks import TASKS
from medical_agentic_rag.observability import trace

T = TypeVar("T", bound=BaseModel)

_client: genai.Client | None = None

MODEL_BY_TIER = {
    "fast": settings.LLM_MODEL_FAST,
    "lite": settings.LLM_MODEL_LITE or settings.LLM_MODEL_FAST,  # chưa cấu hình lite -> dùng fast
    "strong": settings.LLM_MODEL_STRONG,
    "judge": settings.LLM_MODEL_JUDGE or settings.LLM_MODEL_STRONG,  # chưa cấu hình judge -> dùng strong
}


_FENCE_START = re.compile(r"^```[a-zA-Z]*\s*")
_FENCE_END = re.compile(r"\s*```$")


def strip_code_fence(text: str) -> str:
    """Model đôi khi bọc JSON trong ```json ... ``` dù đã yêu cầu response_mime_type=application/json."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = _FENCE_END.sub("", _FENCE_START.sub("", text, count=1)).strip()
    return text


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


async def _stream_text(client, spec, contents, cfg, on_token) -> tuple[str, Any]:
    """Stream text ra ngoài qua on_token, trả về (toàn văn, usage_metadata của mảnh cuối)."""
    parts: list[str] = []
    usage = None
    stream = await client.aio.models.generate_content_stream(
        model=MODEL_BY_TIER[spec.model_tier], contents=contents, config=cfg,
    )
    async for chunk in stream:
        usage = getattr(chunk, "usage_metadata", None) or usage
        piece = chunk.text or ""
        if piece:
            parts.append(piece)
            await on_token(piece)
    return "".join(parts), usage


async def run_task(
    task: str, contents: Any, state: dict,
    schema: Optional[type[T]] = None, tools: Optional[list] = None,
    on_token: Optional[Callable[[str], Awaitable[None]]] = None,
) -> T | str:
    """tools != None (research_agent): trả về response gốc (caller đọc .function_calls).

    on_token != None: gọi generate_content_stream và await on_token(chunk) cho từng mảnh text,
    vẫn trả về chuỗi đầy đủ. Chỉ dùng cho task sinh văn bản tự do (generate_answer) — không
    dùng với schema vì JSON phải parse trọn vẹn.
    """
    spec = TASKS[task]
    client = get_client()
    cfg = types.GenerateContentConfig(
        temperature=spec.temperature,
        max_output_tokens=spec.max_output_tokens,
        response_mime_type="application/json" if schema else None,
        response_schema=schema,
        tools=tools,
        # thinking tokens (gemini-2.5) trừ vào chính max_output_tokens — từng làm
        # cắt cụt JSON và research_agent trả rỗng giữa vòng lặp. Mức thinking cấu
        # hình riêng theo từng task (xem llm/tasks.py) thay vì tắt toàn cục.
        thinking_config=types.ThinkingConfig(thinking_budget=spec.thinking_budget),
    )
    last_exc: Exception | None = None
    for attempt in range(2):  # tối đa 1 lần thử lại
        budget.charge_llm(state, task)
        t0 = time.perf_counter()
        try:
            if on_token is not None:
                out, usage = await _stream_text(client, spec, contents, cfg, on_token)
                resp = None
            else:
                resp = await client.aio.models.generate_content(
                    model=MODEL_BY_TIER[spec.model_tier], contents=contents, config=cfg,
                )
                if tools:
                    out = resp
                else:
                    out = schema.model_validate_json(strip_code_fence(resp.text)) if schema else (resp.text or "")
                usage = getattr(resp, "usage_metadata", None)
            trace.log_llm(
                state, task, attempt, t0, ok=True,
                tokens_in=getattr(usage, "prompt_token_count", None),
                tokens_out=getattr(usage, "candidates_token_count", None),
            )
            return out
        except (ValidationError, errors.APIError, httpx.TransportError) as exc:
            last_exc = exc
            trace.log_llm(state, task, attempt, t0, ok=False, error=type(exc).__name__, detail=str(exc)[:400])
    raise LLMTaskFailed(f"{task}: {last_exc}")
