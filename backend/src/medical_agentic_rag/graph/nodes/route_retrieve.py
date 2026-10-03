"""Node retrieval riêng theo question_type (mục 8.3).

symptom_to_condition dùng chung hybrid_retrieve/rerank của Phase 1+2 (retrieve.py),
không cần node riêng — xem routing.py để biết cạnh nối.
"""
from __future__ import annotations

from medical_agentic_rag.config import settings
from medical_agentic_rag.graph.state import State
from medical_agentic_rag.retrieval.article_index import find_article
from medical_agentic_rag.retrieval.hybrid import retrieve
from medical_agentic_rag.retrieval.rerank import rerank_and_select


async def retrieve_definition(state: State) -> dict:
    """Fast path: hybrid -> rerank; top-1 đủ tự tin thì đánh dấu để bỏ qua grader."""
    candidates = retrieve(state["standalone_question"])
    pool = {c["chunk_id"]: c for c in candidates}
    chunks = rerank_and_select(state["standalone_question"], list(pool.values()))

    confident = bool(
        chunks and settings.RERANK_CONFIDENT_SCORE is not None
        and chunks[0].get("rerank_score", 0.0) >= settings.RERANK_CONFIDENT_SCORE
    )
    return {"retrieved_pool": pool, "reranked_chunks": chunks, "confident": confident}


async def retrieve_section_lookup(state: State) -> dict:
    """find_article (code, không LLM) -> hybrid lọc article_id + section_type; rỗng thì bỏ lọc."""
    entities = state.get("entities") or []
    name = entities[0] if entities else state["standalone_question"]
    match = find_article(name)

    target_types = state.get("target_section_types") or []
    section_type = target_types[0] if target_types else None

    if match:
        candidates = retrieve(state["standalone_question"], article_id=match["article_id"], section_type=section_type)
        if not candidates and section_type:
            candidates = retrieve(state["standalone_question"], article_id=match["article_id"])
    else:
        candidates = retrieve(state["standalone_question"])

    pool = {c["chunk_id"]: c for c in candidates}
    chunks = rerank_and_select(state["standalone_question"], list(pool.values()))
    return {"retrieved_pool": pool, "reranked_chunks": chunks}
