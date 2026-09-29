"""Client google-genai trỏ tới endpoint shopaikey.

Phase 1 chỉ cần một task: generate_answer (text thuần, có trích dẫn [n]).
Retry/budget/schema đầy đủ cho các task khác được thêm ở Phase 2 (module llm/
đã tách sẵn để mở rộng mà không phải đổi cách gọi từ node/graph).
"""
from __future__ import annotations

import asyncio

from google import genai
from google.genai import errors, types

from medical_agentic_rag.config import settings

_client: genai.Client | None = None


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


async def generate_answer(prompt: str, retries: int = 2) -> str:
    client = get_client()
    cfg = types.GenerateContentConfig(temperature=0.2, max_output_tokens=1024)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = await client.aio.models.generate_content(
                model=settings.LLM_MODEL_STRONG, contents=prompt, config=cfg,
            )
            return resp.text or ""
        except errors.APIError as exc:
            last_exc = exc
            await asyncio.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"generate_answer failed after {retries} attempts: {last_exc}")
