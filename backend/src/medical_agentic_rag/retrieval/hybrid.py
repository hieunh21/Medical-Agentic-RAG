"""Hybrid retrieval: dense + sparse tiếng Việt, RRF chạy trong Qdrant."""
from __future__ import annotations

from typing import List

from ingestion.embed_client import embed_query
from ingestion.qdrant_store import get_client, hybrid_query
from ingestion.vi_sparse import query_sparse

from medical_agentic_rag.config import settings


def retrieve(question: str, limit: int | None = None) -> List[dict]:
    dense_vec = embed_query(question)
    sparse_vec = query_sparse(question)
    client = get_client()
    points = hybrid_query(
        client, dense_vec, sparse_vec,
        limit=limit or settings.FUSION_TOP_N,
        prefetch_limit=settings.DENSE_PREFETCH,
    )
    return [{"score": p.score, **p.payload} for p in points]
