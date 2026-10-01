"""Hybrid retrieval: dense + sparse tiếng Việt, RRF chạy trong Qdrant."""
from __future__ import annotations

from typing import List, Optional

from qdrant_client import models

from ingestion.embed_client import embed_query
from ingestion.qdrant_store import get_client, hybrid_query
from ingestion.vi_sparse import query_sparse

from medical_agentic_rag.config import settings


def retrieve(question: str, limit: int | None = None, section_type: Optional[str] = None) -> List[dict]:
    dense_vec = embed_query(question)
    sparse_vec = query_sparse(question)
    client = get_client()
    query_filter = None
    if section_type:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="section_type", match=models.MatchValue(value=section_type))],
        )
    points = hybrid_query(
        client, dense_vec, sparse_vec,
        limit=limit or settings.FUSION_TOP_N,
        prefetch_limit=settings.DENSE_PREFETCH,
        query_filter=query_filter,
    )
    return [{"score": p.score, **p.payload} for p in points]
