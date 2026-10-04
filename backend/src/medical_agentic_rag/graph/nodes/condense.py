"""Node condense_question — biến câu nối tiếp thành câu hỏi độc lập (mục 8.1).

Bỏ qua lời gọi LLM nếu đây là lượt đầu (chưa có lịch sử).
"""
from __future__ import annotations

from medical_agentic_rag.budget import degrade_on_llm_error
from medical_agentic_rag.config import settings
from medical_agentic_rag.graph.state import State
from medical_agentic_rag.llm.client import run_task
from medical_agentic_rag.llm.prompts import build_condense_prompt
from medical_agentic_rag.llm.schemas import Condensed
from medical_agentic_rag.retrieval.article_index import get_titles


@degrade_on_llm_error(lambda s: {"standalone_question": s["question"], "is_followup": False})
async def condense_question(state: State) -> dict:
    messages = state.get("messages") or []
    history = messages[:-1]  # bỏ câu hỏi mới nhất (chính nó) ra khỏi lịch sử
    if not history:
        return {"standalone_question": state["question"], "is_followup": False}

    active_titles = get_titles(state.get("active_article_ids") or [])
    prompt = build_condense_prompt(history[-settings.HISTORY_TURNS * 2 :], state["question"], active_titles)
    condensed: Condensed = await run_task("condense_question", prompt, state, schema=Condensed)
    return {
        "standalone_question": condensed.standalone_question,
        "is_followup": condensed.is_followup,
        "llm_calls": state["llm_calls"],  # run_task() đã mutate state["llm_calls"] in-place
    }
