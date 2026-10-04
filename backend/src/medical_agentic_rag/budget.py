"""Ngân sách lời gọi LLM — code giữ, không để LLM tự quyết định dừng khi nào.

Lời gọi cuối cùng (generate_answer) luôn được chừa chỗ: các task khác chỉ được dùng
tối đa MAX-1 lượt, nên khi ngân sách cạn vẫn còn 1 lượt để viết câu trả lời từ evidence
đã có. Node nào có thể chạm trần hoặc gặp LLM lỗi thì bọc `degrade_on_llm_error` để thoái hoá êm thay vì
làm sập cả request.
"""
from __future__ import annotations

import functools
from typing import Awaitable, Callable

from medical_agentic_rag.config import settings
from medical_agentic_rag.errors import LLMTaskFailed

FINAL_TASK = "generate_answer"


class BudgetExceeded(Exception):
    pass


def _limit(task: str) -> int:
    return settings.MAX_LLM_CALLS_PER_REQUEST if task == FINAL_TASK else settings.MAX_LLM_CALLS_PER_REQUEST - 1


def charge_llm(state: dict, task: str) -> None:
    calls = state.get("llm_calls", 0)
    if calls >= _limit(task):
        raise BudgetExceeded(task)
    state["llm_calls"] = calls + 1


def exhausted(state: dict) -> bool:
    """True khi không còn lượt cho task nào ngoài generate_answer."""
    return state.get("llm_calls", 0) >= settings.MAX_LLM_CALLS_PER_REQUEST - 1


def degrade_on_llm_error(fallback: Callable[[dict], dict]):
    """Bọc node async: hết ngân sách hoặc LLM lỗi (LLMTaskFailed) thì trả `fallback(state)`
    thay vì làm sập request."""
    def deco(node: Callable[[dict], Awaitable[dict]]):
        @functools.wraps(node)
        async def wrapper(state: dict) -> dict:
            try:
                return await node(state)
            except (BudgetExceeded, LLMTaskFailed):
                return {**fallback(state), "llm_calls": state.get("llm_calls", 0)}
        return wrapper
    return deco
