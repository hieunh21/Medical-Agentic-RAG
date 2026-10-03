"""Qdrant: 1 collection, 2 named vector (dense + sparse), RRF chạy trong Qdrant."""
from __future__ import annotations

import os
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient, models

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "youmed_medical")
DENSE_DIM = 768

PAYLOAD_KEYWORD_FIELDS = ["article_id", "category", "section_type", "chunk_type"]


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def create_collection(client: QdrantClient, recreate: bool = False) -> None:
    if recreate and client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    if client.collection_exists(QDRANT_COLLECTION):
        return
    client.create_collection(
        QDRANT_COLLECTION,
        vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field in PAYLOAD_KEYWORD_FIELDS:
        client.create_payload_index(QDRANT_COLLECTION, field, models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(QDRANT_COLLECTION, "updated_date", models.PayloadSchemaType.DATETIME)


def upsert_chunks(
    client: QdrantClient,
    chunks: List[dict],
    dense_vecs: List[List[float]],
    sparse_vecs: List[tuple[list[int], list[float]]],
    batch_size: int = 64,
) -> None:
    points = []
    for chunk, dense, (s_idx, s_val) in zip(chunks, dense_vecs, sparse_vecs):
        points.append(models.PointStruct(
            id=point_id(chunk["chunk_id"]),
            vector={
                "dense": dense,
                "sparse": models.SparseVector(indices=s_idx, values=s_val),
            },
            payload=chunk,
        ))
    for i in range(0, len(points), batch_size):
        client.upsert(QDRANT_COLLECTION, points=points[i : i + batch_size])


def get_chunks_by_article(client: QdrantClient, article_id: str, limit: int = 50) -> List[dict]:
    """Toàn bộ chunk của 1 bài — dùng cho get_article_outline() của research agent."""
    points, _ = client.scroll(
        QDRANT_COLLECTION, limit=limit, with_payload=True, with_vectors=False,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="article_id", match=models.MatchValue(value=article_id))],
        ),
    )
    return [p.payload for p in points]


def get_chunks_by_ids(client: QdrantClient, chunk_ids: List[str]) -> List[dict]:
    """Đọc nguyên văn một số chunk_id cụ thể — dùng cho read_chunks() của research agent."""
    points = client.retrieve(QDRANT_COLLECTION, ids=[point_id(c) for c in chunk_ids], with_payload=True)
    return [p.payload for p in points]


def hybrid_query(
    client: QdrantClient,
    dense_vec: List[float],
    sparse_vec: tuple[list[int], list[float]],
    limit: int = 30,
    prefetch_limit: int = 40,
    query_filter: Optional[models.Filter] = None,
):
    s_idx, s_val = sparse_vec
    return client.query_points(
        QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=prefetch_limit, filter=query_filter),
            models.Prefetch(
                query=models.SparseVector(indices=s_idx, values=s_val),
                using="sparse", limit=prefetch_limit, filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
    ).points
