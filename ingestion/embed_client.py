"""HTTP client gọi model server (:8001) để lấy dense embedding và điểm rerank."""
from __future__ import annotations

import os
from typing import List, Optional

import requests

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8001")


def embed_documents(texts: List[str], titles: Optional[List[str]] = None) -> List[List[float]]:
    resp = requests.post(
        f"{MODEL_SERVER_URL}/embed",
        json={"texts": texts, "task": "document", "titles": titles},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def embed_query(text: str) -> List[float]:
    resp = requests.post(
        f"{MODEL_SERVER_URL}/embed",
        json={"texts": [text], "task": "query"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def rerank(query: str, documents: List[str]) -> List[float]:
    if not documents:
        return []
    resp = requests.post(
        f"{MODEL_SERVER_URL}/rerank",
        json={"query": query, "documents": documents},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["scores"]
