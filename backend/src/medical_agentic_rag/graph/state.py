"""State cho graph Phase 1+2. Các field hội thoại/routing/agent sẽ thêm ở Phase 3."""
from __future__ import annotations

from typing import Optional, TypedDict


class State(TypedDict, total=False):
    # câu hỏi
    question: str
    standalone_question: str
    aspects: list[str]

    # retrieval
    retrieved_pool: dict[str, dict]  # chunk_id -> chunk payload, tích lũy qua các lượt sửa
    reranked_chunks: list[dict]      # context hiện tại, đã rerank theo câu hỏi gốc
    corrections: int

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


def init_state(question: str, trace_id: Optional[str] = None) -> State:
    return {
        "question": question,
        "standalone_question": question,
        "aspects": [DEFAULT_ASPECT],
        "retrieved_pool": {},
        "corrections": 0,
        "llm_calls": 0,
        "trace_id": trace_id,
    }
