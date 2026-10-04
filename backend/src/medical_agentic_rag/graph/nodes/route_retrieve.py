"""Node retrieval riêng theo question_type (mục 8.3).

symptom_to_condition dùng chung hybrid_retrieve/rerank của Phase 1+2 (retrieve.py),
không cần node riêng — xem routing.py để biết cạnh nối.
"""
from __future__ import annotations

import asyncio

from medical_agentic_rag.config import settings
from medical_agentic_rag.graph.state import State
from medical_agentic_rag.retrieval.article_index import find_article
from medical_agentic_rag.retrieval.hybrid import retrieve
from medical_agentic_rag.retrieval.rerank import rerank_and_select


async def retrieve_definition(state: State) -> dict:
    """Fast path: hybrid -> rerank; top-1 đủ tự tin thì đánh dấu để bỏ qua grader."""
    candidates = await asyncio.to_thread(retrieve, state["standalone_question"])
    pool = {c["chunk_id"]: c for c in candidates}
    chunks = await asyncio.to_thread(rerank_and_select, state["standalone_question"], list(pool.values()))

    confident = bool(
        chunks and settings.RERANK_CONFIDENT_SCORE is not None
        and chunks[0].get("rerank_score", 0.0) >= settings.RERANK_CONFIDENT_SCORE
    )
    return {"retrieved_pool": pool, "reranked_chunks": chunks, "confident": confident}


def _section_lookup_sync(state: State, widen: bool = True) -> tuple[dict, list[dict]]:
    """Ứng viên cho section_lookup: kết quả lọc theo bài đã khớp + (widen) kết quả không lọc.

    find_article chỉ khớp mờ ra 1 bài, trong khi corpus có nhiều bài cùng một bệnh (vd. 4 bài
    tăng huyết áp, 2 bài cúm mùa): khoá cứng vào 1 bài làm mất bài đúng khi nó là bài "anh em".
    Gộp thêm ứng viên không lọc để reranker chọn giữa các bài. widen=False là hành vi cũ
    (chỉ dùng cho eval/run_section_lookup_compare.py).
    """
    question = state["standalone_question"]
    entities = state.get("entities") or []
    name = entities[0] if entities else question
    match = find_article(name)

    target_types = state.get("target_section_types") or []
    section_type = target_types[0] if target_types else None

    focused: list[dict] = []
    if match:
        focused = retrieve(question, article_id=match["article_id"], section_type=section_type)
        if not focused and section_type:
            focused = retrieve(question, article_id=match["article_id"])
    broad = retrieve(question) if (widen or not match) else []

    pool = {c["chunk_id"]: c for c in focused}
    for c in broad:
        pool.setdefault(c["chunk_id"], c)
    return pool, rerank_and_select(question, list(pool.values()))


async def retrieve_section_lookup(state: State) -> dict:
    """find_article (code, không LLM) -> ứng viên lọc theo bài + ứng viên không lọc -> rerank."""
    pool, chunks = await asyncio.to_thread(_section_lookup_sync, state)
    return {"retrieved_pool": pool, "reranked_chunks": chunks}
