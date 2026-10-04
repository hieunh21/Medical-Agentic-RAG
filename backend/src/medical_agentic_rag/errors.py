"""Exception dùng chung (tách riêng để budget.py và llm/client.py không import vòng nhau)."""
from __future__ import annotations


class LLMTaskFailed(RuntimeError):
    """run_task() thất bại sau khi đã thử lại (API lỗi, timeout, JSON sai schema)."""
