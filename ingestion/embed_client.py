"""Client lấy dense embedding và điểm rerank — hai chế độ, chọn bằng biến môi trường:

    MODEL_SPACE đặt (vd "hhieu21/medrag-models")  -> gọi Hugging Face ZeroGPU Space qua gradio_client.
                                                    Dùng khi deploy: backend không cần torch / GPU.
    MODEL_SPACE trống                             -> gọi model server local (:8001) qua HTTP.
                                                    Dùng khi dev và khi ingest (embed 6k chunk tốn quota GPU).

Space giữ nguyên ý nghĩa API của model_server/server.py (xem deploy/hf_models_space/app.py).
"""
from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from typing import List, Optional

import requests

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:8001")  # không dùng localhost: xem config.py
MODEL_SPACE = os.getenv("MODEL_SPACE", "").strip()

_space_client = None
_space_lock = threading.Lock()


def _space():
    """Một gradio_client.Client dùng chung (kết nối tới Space mất ~2-3s, không tạo lại mỗi lần gọi)."""
    global _space_client
    with _space_lock:
        if _space_client is None:
            from gradio_client import Client  # import muộn: chế độ local không cần gói này

            _space_client = Client(MODEL_SPACE, token=os.getenv("HF_TOKEN") or None, verbose=False)
        return _space_client


def _call_space(api: str, payload: dict) -> dict:
    return json.loads(_space().predict(json.dumps(payload, ensure_ascii=False), api_name=f"/{api}"))


def _post(path: str, payload: dict, timeout: int) -> dict:
    resp = requests.post(f"{MODEL_SERVER_URL}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def embed_documents(texts: List[str], titles: Optional[List[str]] = None) -> List[List[float]]:
    payload = {"texts": texts, "task": "document", "titles": titles}
    out = _call_space("embed", payload) if MODEL_SPACE else _post("/embed", payload, timeout=120)
    return out["embeddings"]


@lru_cache(maxsize=512)
def _embed_query_cached(text: str) -> tuple[float, ...]:
    payload = {"texts": [text], "task": "query"}
    out = _call_space("embed", payload) if MODEL_SPACE else _post("/embed", payload, timeout=30)
    return tuple(out["embeddings"][0])


def embed_query(text: str) -> List[float]:
    """Có cache: section_lookup gọi retrieve() 2-3 lần với cùng câu hỏi (lọc theo bài, không lọc), mỗi lần
    embed lại y hệt. Với ZeroGPU mỗi lời gọi tốn ~2s (hàng đợi + gắn GPU) dù tính toán chỉ ~0.06s."""
    return list(_embed_query_cached(text))


def rerank(query: str, documents: List[str]) -> List[float]:
    if not documents:
        return []
    payload = {"query": query, "documents": documents}
    out = _call_space("rerank", payload) if MODEL_SPACE else _post("/rerank", payload, timeout=60)
    return out["scores"]


def ping() -> dict:
    """Kiểm tra (và đánh thức) nơi chạy model — /ping của Space không dùng GPU."""
    if MODEL_SPACE:
        return _call_space("ping", {})
    resp = requests.get(f"{MODEL_SERVER_URL}/health", timeout=5)
    resp.raise_for_status()
    return resp.json()
