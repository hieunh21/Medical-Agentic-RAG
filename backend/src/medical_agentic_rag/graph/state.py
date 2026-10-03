"""State cho graph Phase 1+2+3 (mục 8.6). An toàn/citation/nguồn ngoài để dành Phase 4+5."""
from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


class State(TypedDict, total=False):
    # hội thoại
    messages: Annotated[list[AnyMessage], add_messages]
    thread_id: str
    active_article_ids: list[str]

    # hiểu câu hỏi
    question: str
    standalone_question: str
    is_followup: bool
    question_type: str
    safety_label: str  # thu thập ở Phase 3, hành vi xử lý thật sự để Phase 5
    entities: list[str]
    aspects: list[str]
    target_section_types: list[str]
    route: str

    # retrieval
    subqueries: list[str]
    sub_section_types: list[Optional[str]]
    sub_results: Annotated[list[dict], operator.add]
    retrieved_pool: dict[str, dict]  # chunk_id -> chunk payload, tích lũy qua các lượt sửa
    reranked_chunks: list[dict]      # context hiện tại, đã rerank theo câu hỏi gốc
    corrections: int
    agent_tool_calls: int
    confident: bool  # fast path của route "definition" — rerank top-1 đủ tự tin thì bỏ qua grader

    # evidence
    n_relevant: int
    coverage: float
    missing_aspects: list[str]
    evidence_status: str             # sufficient | partial | insufficient

    # trả lời
    final_answer: str
    sources: list[dict]

    # kiểm soát
    llm_calls: int
    trace_id: Optional[str]


DEFAULT_ASPECT = "trả lời trực tiếp câu hỏi"


def init_state(question: str, thread_id: Optional[str] = None, trace_id: Optional[str] = None) -> State:
    """Input cho MỖI lượt hỏi (kể cả lượt đầu lẫn lượt nối tiếp trong cùng thread_id).

    Cố ý KHÔNG đặt "active_article_ids": nếu thread_id đã có checkpoint, LangGraph giữ
    nguyên giá trị cũ của field không có reducer khi key đó vắng mặt trong input; nếu là
    thread mới thì field này vốn không tồn tại, các node đọc bằng `.get(..) or []`.
    """
    return {
        "messages": [HumanMessage(content=question)],
        "thread_id": thread_id or "",
        "question": question,
        "standalone_question": question,
        "is_followup": False,
        "aspects": [DEFAULT_ASPECT],
        "retrieved_pool": {},
        "corrections": 0,
        "agent_tool_calls": 0,
        "llm_calls": 0,
        "trace_id": trace_id,
    }
