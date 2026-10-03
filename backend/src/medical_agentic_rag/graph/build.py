"""Lắp StateGraph Phase 1+2+3: condense -> analyze -> route theo question_type ->
retrieve (khác nhau theo route) -> grade -> (sửa)* -> trả lời.

Checkpointer (SQLite) truyền từ ngoài vào vì AsyncSqliteSaver là 1 async context
manager cần giữ mở xuyên suốt vòng đời app — xem cli.py / api/debug_app.py.
"""
from __future__ import annotations

from typing import Optional

from langgraph.graph import END, StateGraph

from medical_agentic_rag.agent.loop import research_agent
from medical_agentic_rag.graph.nodes.analyze import analyze_query
from medical_agentic_rag.graph.nodes.condense import condense_question
from medical_agentic_rag.graph.nodes.correct import rewrite_query, targeted_retrieve
from medical_agentic_rag.graph.nodes.generate import (
    generate_answer_node, no_info_response_node, out_of_scope_response_node,
)
from medical_agentic_rag.graph.nodes.grade import grade_evidence
from medical_agentic_rag.graph.nodes.retrieve import hybrid_retrieve, rerank_node
from medical_agentic_rag.graph.nodes.route_retrieve import retrieve_definition, retrieve_section_lookup
from medical_agentic_rag.graph.nodes.subquery import fan_out, merge_evidence, plan_subqueries, retrieve_sub
from medical_agentic_rag.graph.routing import after_definition_retrieve, after_grade, route_by_type
from medical_agentic_rag.graph.state import State


def build_graph():
    g = StateGraph(State)

    # hiểu câu hỏi
    g.add_node("condense_question", condense_question)
    g.add_node("analyze_query", analyze_query)

    # retrieval theo route
    g.add_node("retrieve_definition", retrieve_definition)
    g.add_node("retrieve_section_lookup", retrieve_section_lookup)
    g.add_node("hybrid_retrieve", hybrid_retrieve)  # symptom_to_condition dùng chung Phase 1+2
    g.add_node("rerank", rerank_node)
    g.add_node("plan_subqueries", plan_subqueries)
    g.add_node("retrieve_sub", retrieve_sub)
    g.add_node("merge_evidence", merge_evidence)
    g.add_node("research_agent", research_agent)

    # evidence + sửa lỗi (Phase 2, dùng chung cho mọi route)
    g.add_node("grade_evidence", grade_evidence)
    g.add_node("targeted_retrieve", targeted_retrieve)
    g.add_node("rewrite_query", rewrite_query)

    # trả lời
    g.add_node("generate_answer", generate_answer_node)
    g.add_node("no_info_response", no_info_response_node)
    g.add_node("out_of_scope_response", out_of_scope_response_node)

    g.set_entry_point("condense_question")
    g.add_edge("condense_question", "analyze_query")
    g.add_conditional_edges(
        "analyze_query", route_by_type,
        {
            "retrieve_definition": "retrieve_definition",
            "retrieve_section_lookup": "retrieve_section_lookup",
            "hybrid_retrieve": "hybrid_retrieve",
            "plan_subqueries": "plan_subqueries",
            "research_agent": "research_agent",
            "out_of_scope_response": "out_of_scope_response",
        },
    )

    # definition: fast path — rerank top-1 đủ tự tin thì bỏ qua grader
    g.add_conditional_edges(
        "retrieve_definition", after_definition_retrieve,
        {"generate_answer": "generate_answer", "grade_evidence": "grade_evidence"},
    )
    g.add_edge("retrieve_section_lookup", "grade_evidence")
    g.add_edge("hybrid_retrieve", "rerank")
    g.add_edge("rerank", "grade_evidence")

    # comparison / multi_aspect: retrieve song song bằng Send
    g.add_conditional_edges("plan_subqueries", fan_out, ["retrieve_sub"])
    g.add_edge("retrieve_sub", "merge_evidence")
    g.add_edge("merge_evidence", "grade_evidence")

    # complex: research agent chọn evidence, vẫn rerank lại theo câu hỏi gốc trước khi chấm
    g.add_edge("research_agent", "rerank")

    # vòng sửa lỗi (Phase 2)
    g.add_conditional_edges(
        "grade_evidence", after_grade,
        {
            "generate_answer": "generate_answer",
            "targeted_retrieve": "targeted_retrieve",
            "rewrite_query": "rewrite_query",
            "no_info_response": "no_info_response",
            "external_fallback": "no_info_response",  # Phase 4 (PubMed) chưa có node riêng
        },
    )
    g.add_edge("targeted_retrieve", "grade_evidence")
    g.add_edge("rewrite_query", "grade_evidence")

    g.add_edge("generate_answer", END)
    g.add_edge("no_info_response", END)
    g.add_edge("out_of_scope_response", END)
    return g


def compile_graph(checkpointer: Optional[object] = None):
    return build_graph().compile(checkpointer=checkpointer)
