"""Lắp StateGraph Phase 1+2+3+5: safety_rules -> condense -> analyze -> route theo question_type ->
retrieve (khác nhau theo route) -> grade -> (sửa)* -> generate -> citation_validator.

Checkpointer (SQLite) truyền từ ngoài vào vì AsyncSqliteSaver là 1 async context
manager cần giữ mở xuyên suốt vòng đời app — xem cli.py / api/debug_app.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langgraph.graph import END, StateGraph

from medical_agentic_rag.agent.loop import research_agent
from medical_agentic_rag.graph.nodes.analyze import analyze_query
from medical_agentic_rag.graph.nodes.condense import condense_question
from medical_agentic_rag.graph.nodes.correct import rewrite_query, targeted_retrieve
from medical_agentic_rag.graph.nodes.generate import (
    generate_answer_node, no_info_response_node, out_of_scope_response_node, safety_response_node,
)
from medical_agentic_rag.graph.nodes.grade import grade_evidence
from medical_agentic_rag.graph.nodes.retrieve import hybrid_retrieve, rerank_node
from medical_agentic_rag.graph.nodes.safety import safety_rules
from medical_agentic_rag.graph.nodes.route_retrieve import retrieve_definition, retrieve_section_lookup
from medical_agentic_rag.graph.nodes.subquery import fan_out, merge_evidence, plan_subqueries, retrieve_sub
from medical_agentic_rag.graph.nodes.validate import finalize_answer, validate_citations
from medical_agentic_rag.graph.routing import (
    after_definition_retrieve, after_grade, after_safety_rules, after_validate, route_by_type,
)
from medical_agentic_rag.graph.state import State
from medical_agentic_rag.observability.trace import traced


@dataclass(frozen=True)
class Features:
    """Bật / tắt từng thành phần để chạy ablation (eval/run_ablation.py). Mặc định: bật hết.

    routing    — condense + analyze + route theo question_type (tắt: mọi câu đi hybrid -> rerank).
    corrective — grade_evidence + vòng sửa lỗi (tắt: rerank xong generate luôn).
    agent      — route "complex" dùng research_agent (tắt: rơi về hybrid_retrieve).
    safety     — safety_rules + nhãn safety của analyze (dừng pipeline, ràng buộc generate).
    validator  — citation_validator (tắt: chốt bản nháp nguyên trạng).
    generate   — tắt thì graph dừng ngay sau retrieval / grade (không sinh câu trả lời); chỉ dùng để
                 đo retrieval rẻ hơn (eval/run_ablation.py --retrieval-only).
    """
    routing: bool = True
    corrective: bool = True
    agent: bool = True
    safety: bool = True
    validator: bool = True
    generate: bool = True

    def __post_init__(self):
        if self.safety and not self.routing:
            raise ValueError("safety lớp 2 nằm trong analyze_query: safety=True cần routing=True")


def _without_safety_label(node):
    """Giữ analyze_query nhưng bỏ nhãn safety (cấu hình safety=False)."""
    async def wrapper(state):
        return {**await node(state), "safety_label": "normal"}
    wrapper.__name__ = node.__name__
    return wrapper


def build_graph(features: Features = Features()):
    g = StateGraph(State)

    def add(name: str, fn) -> None:
        g.add_node(name, traced(name, fn))

    # safety lớp 1 (rule) rồi hiểu câu hỏi
    add("safety_rules", safety_rules)
    add("condense_question", condense_question)
    add("analyze_query", analyze_query if features.safety else _without_safety_label(analyze_query))

    # retrieval theo route
    add("retrieve_definition", retrieve_definition)
    add("retrieve_section_lookup", retrieve_section_lookup)
    add("hybrid_retrieve", hybrid_retrieve)  # symptom_to_condition dùng chung Phase 1+2
    add("rerank", rerank_node)
    add("plan_subqueries", plan_subqueries)
    add("retrieve_sub", retrieve_sub)
    add("merge_evidence", merge_evidence)
    add("research_agent", research_agent)

    # evidence + sửa lỗi (Phase 2, dùng chung cho mọi route)
    add("grade_evidence", grade_evidence)
    add("targeted_retrieve", targeted_retrieve)
    add("rewrite_query", rewrite_query)

    # trả lời
    add("generate_answer", generate_answer_node)
    add("no_info_response", no_info_response_node)
    add("out_of_scope_response", out_of_scope_response_node)
    add("safety_response", safety_response_node)
    add("citation_validator", validate_citations if features.validator else finalize_answer)

    to_generate = "generate_answer" if features.generate else END
    to_grade = "grade_evidence" if features.corrective else to_generate

    # đầu vào: safety_rules -> condense -> analyze -> route; thiếu thành phần nào thì nối tắt
    if features.routing:
        entry = "safety_rules" if features.safety else "condense_question"
        g.set_entry_point(entry)
        if features.safety:
            g.add_conditional_edges(
                "safety_rules", after_safety_rules,
                {"safety_response": "safety_response", "condense_question": "condense_question"},
            )
        g.add_edge("condense_question", "analyze_query")
        routes = {
            "retrieve_definition": "retrieve_definition",
            "retrieve_section_lookup": "retrieve_section_lookup",
            "hybrid_retrieve": "hybrid_retrieve",
            "plan_subqueries": "plan_subqueries",
            "research_agent": "research_agent" if features.agent else "hybrid_retrieve",
            "out_of_scope_response": "out_of_scope_response",
        }
        if features.safety:
            routes["safety_response"] = "safety_response"
        g.add_conditional_edges("analyze_query", route_by_type, routes)
    else:
        g.set_entry_point("hybrid_retrieve")

    # definition: fast path — rerank top-1 đủ tự tin thì bỏ qua grader
    g.add_conditional_edges(
        "retrieve_definition", after_definition_retrieve,
        {"generate_answer": to_generate, "grade_evidence": to_grade},
    )
    g.add_edge("retrieve_section_lookup", to_grade)
    g.add_edge("hybrid_retrieve", "rerank")
    g.add_edge("rerank", to_grade)

    # comparison / multi_aspect: retrieve song song bằng Send
    g.add_conditional_edges("plan_subqueries", fan_out, ["retrieve_sub"])
    g.add_edge("retrieve_sub", "merge_evidence")
    g.add_edge("merge_evidence", to_grade)

    # complex: research agent chọn evidence, vẫn rerank lại theo câu hỏi gốc trước khi chấm
    g.add_edge("research_agent", "rerank")

    # vòng sửa lỗi (Phase 2)
    if features.corrective:
        g.add_conditional_edges(
            "grade_evidence", after_grade,
            {
                "generate_answer": to_generate,
                "targeted_retrieve": "targeted_retrieve",
                "rewrite_query": "rewrite_query",
                "no_info_response": "no_info_response",
            },
        )
        g.add_edge("targeted_retrieve", "grade_evidence")
        g.add_edge("rewrite_query", "grade_evidence")

    # generate -> kiểm tra trích dẫn (viết lại tối đa 1 lần) -> kết thúc
    g.add_edge("generate_answer", "citation_validator")
    if features.validator:
        g.add_conditional_edges(
            "citation_validator", after_validate, {"generate_answer": "generate_answer", "end": END},
        )
    else:
        g.add_edge("citation_validator", END)
    g.add_edge("no_info_response", END)
    g.add_edge("out_of_scope_response", END)
    g.add_edge("safety_response", END)
    return g


def compile_graph(checkpointer: Optional[object] = None, features: Features = Features()):
    return build_graph(features).compile(checkpointer=checkpointer)
