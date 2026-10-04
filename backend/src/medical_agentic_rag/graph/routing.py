"""Routing sau analyze_query (mục 8.3) và luật dừng sau grade_evidence (mục 7.4) —
viết bằng code, không để LLM tự quyết."""
from __future__ import annotations

from medical_agentic_rag import budget
from medical_agentic_rag.config import settings
from medical_agentic_rag.graph.state import State
from medical_agentic_rag.safety.rules import STOP_LABELS

ROUTE_BY_QUESTION_TYPE = {
    "definition": "retrieve_definition",
    "section_lookup": "retrieve_section_lookup",
    "symptom_to_condition": "hybrid_retrieve",
    # comparison: eval/run_routing_compare.py cho thấy subquery (0.833 article_coverage)
    # KHÔNG vượt hybrid đơn giản (0.972) trên corpus này — 2 thực thể so sánh thường đã
    # nêu tên rõ, hybrid một lượt đã đủ tìm đúng cả 2 bài. Route về baseline đơn giản hơn.
    "comparison": "hybrid_retrieve",
    "multi_aspect": "plan_subqueries",  # subquery thắng rõ ở đây: section_coverage 0.667 vs 0.333
    "complex": "research_agent",
    "out_of_scope": "out_of_scope_response",
}


def after_safety_rules(state: State) -> str:
    if state.get("safety_label") in STOP_LABELS and not state.get("safety_deferred"):
        return "safety_response"
    return "condense_question"


def after_validate(state: State) -> str:
    return "generate_answer" if state.get("regenerate") else "end"


def route_by_type(state: State) -> str:
    if state.get("safety_label") in STOP_LABELS:  # lớp 2 (analyze_query) bắt ca rule bỏ sót
        return "safety_response"
    route = ROUTE_BY_QUESTION_TYPE.get(state.get("question_type", ""), "hybrid_retrieve")
    if route == "research_agent" and not settings.AGENT_ENABLED:
        return "hybrid_retrieve"  # RESEARCH_AGENT_ENABLED=false -> fallback đường rẻ
    return route


def after_definition_retrieve(state: State) -> str:
    return "generate_answer" if state.get("confident") else "grade_evidence"


def after_grade(state: State) -> str:
    if budget.exhausted(state):
        return "generate_answer" if state["n_relevant"] > 0 else "no_info_response"
    if state.get("grade_failed") and state["n_relevant"] > 0:
        return "generate_answer"
    if state["n_relevant"] == 0:
        if state["corrections"] < settings.MAX_CORRECTIONS:
            return "rewrite_query"
        return "no_info_response"
    if state["coverage"] >= settings.COVERAGE_THRESHOLD:
        return "generate_answer"
    if state["corrections"] < settings.MAX_CORRECTIONS:
        return "targeted_retrieve"
    return "generate_answer"  # evidence_status = partial
