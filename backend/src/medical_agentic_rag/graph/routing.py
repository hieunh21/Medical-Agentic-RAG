"""Luật dừng sau grade_evidence — viết bằng code, không để LLM tự quyết (mục 7.4)."""
from __future__ import annotations

from medical_agentic_rag import budget
from medical_agentic_rag.config import settings
from medical_agentic_rag.graph.state import State


def after_grade(state: State) -> str:
    if budget.exhausted(state):
        return "generate_answer" if state["n_relevant"] > 0 else "no_info_response"
    if state["n_relevant"] == 0:
        if state["corrections"] < settings.MAX_CORRECTIONS:
            return "rewrite_query"
        return "external_fallback" if settings.FALLBACK_ENABLED else "no_info_response"
    if state["coverage"] >= settings.COVERAGE_THRESHOLD:
        return "generate_answer"
    if state["corrections"] < settings.MAX_CORRECTIONS:
        return "targeted_retrieve"
    return "external_fallback" if settings.FALLBACK_ENABLED else "generate_answer"  # evidence_status = partial
