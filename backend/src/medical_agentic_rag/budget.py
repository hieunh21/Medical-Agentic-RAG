"""Ngân sách lời gọi LLM — code giữ, không để LLM tự quyết định dừng khi nào."""
from __future__ import annotations

from medical_agentic_rag.config import settings


class BudgetExceeded(Exception):
    pass


def charge_llm(state: dict, task: str) -> None:
    state["llm_calls"] = state.get("llm_calls", 0) + 1
    if state["llm_calls"] > settings.MAX_LLM_CALLS_PER_REQUEST:
        raise BudgetExceeded(task)


def exhausted(state: dict) -> bool:
    return state.get("llm_calls", 0) >= settings.MAX_LLM_CALLS_PER_REQUEST
