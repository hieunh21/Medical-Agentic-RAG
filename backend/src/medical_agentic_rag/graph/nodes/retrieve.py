"""Node: hybrid_retrieve + rerank — lượt retrieval đầu tiên."""
from __future__ import annotations

from medical_agentic_rag.graph.state import State
from medical_agentic_rag.retrieval.hybrid import retrieve
from medical_agentic_rag.retrieval.rerank import rerank_and_select


def hybrid_retrieve(state: State) -> dict:
    candidates = retrieve(state["standalone_question"])
    pool = {c["chunk_id"]: c for c in candidates}
    return {"retrieved_pool": pool}


def rerank_node(state: State) -> dict:
    pool = state["retrieved_pool"]
    chunks = rerank_and_select(state["standalone_question"], list(pool.values()))
    return {"reranked_chunks": chunks}
