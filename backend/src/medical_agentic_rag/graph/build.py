"""Lắp StateGraph cho Phase 1+2: retrieve -> rerank -> grade -> (sửa)* -> trả lời."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from medical_agentic_rag.graph.nodes.correct import rewrite_query, targeted_retrieve
from medical_agentic_rag.graph.nodes.generate import generate_answer_node, no_info_response_node
from medical_agentic_rag.graph.nodes.grade import grade_evidence
from medical_agentic_rag.graph.nodes.retrieve import hybrid_retrieve, rerank_node
from medical_agentic_rag.graph.routing import after_grade
from medical_agentic_rag.graph.state import State


def build_graph():
    g = StateGraph(State)
    g.add_node("hybrid_retrieve", hybrid_retrieve)
    g.add_node("rerank", rerank_node)
    g.add_node("grade_evidence", grade_evidence)
    g.add_node("targeted_retrieve", targeted_retrieve)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate_answer", generate_answer_node)
    g.add_node("no_info_response", no_info_response_node)

    g.set_entry_point("hybrid_retrieve")
    g.add_edge("hybrid_retrieve", "rerank")
    g.add_edge("rerank", "grade_evidence")
    g.add_conditional_edges(
        "grade_evidence",
        after_grade,
        {
            "generate_answer": "generate_answer",
            "targeted_retrieve": "targeted_retrieve",
            "rewrite_query": "rewrite_query",
            "no_info_response": "no_info_response",
            # Phase 4 (PubMed) chưa có node riêng — tạm trả "chưa đủ thông tin".
            "external_fallback": "no_info_response",
        },
    )
    g.add_edge("targeted_retrieve", "grade_evidence")
    g.add_edge("rewrite_query", "grade_evidence")
    g.add_edge("generate_answer", END)
    g.add_edge("no_info_response", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
