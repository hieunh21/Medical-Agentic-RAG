"""Node trả lời: generate_answer (có evidence) và no_info_response (template, không LLM)."""
from __future__ import annotations

from medical_agentic_rag.answer import build_prompt, format_sources
from medical_agentic_rag.config import settings
from medical_agentic_rag.graph.state import State
from medical_agentic_rag.llm.client import run_task

NO_INFO = "YouMed hiện chưa có bài viết về vấn đề này. Bạn nên hỏi ý kiến bác sĩ để được tư vấn chính xác."


async def generate_answer_node(state: State) -> dict:
    chunks = state["reranked_chunks"]
    status = "sufficient" if state.get("coverage", 0.0) >= settings.COVERAGE_THRESHOLD else "partial"
    prompt = build_prompt(state["standalone_question"], chunks, evidence_status=status)
    answer = await run_task("generate_answer", prompt, state)
    return {
        "final_answer": answer, "sources": format_sources(chunks), "evidence_status": status,
        "llm_calls": state["llm_calls"],  # run_task() đã mutate state["llm_calls"] in-place
    }


def no_info_response_node(state: State) -> dict:
    return {"final_answer": NO_INFO, "sources": [], "evidence_status": "insufficient"}
