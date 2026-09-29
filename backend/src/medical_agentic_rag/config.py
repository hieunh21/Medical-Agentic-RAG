"""Cấu hình đọc từ biến môi trường (.env) — không hard-code secret trong code."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM
    SHOPAIKEY_API_KEY = os.getenv("SHOPAIKEY_API_KEY", "")
    GENAI_BASE_URL = os.getenv("GENAI_BASE_URL", "https://api.shopaikey.com")
    LLM_MODEL_FAST = os.getenv("LLM_MODEL_FAST", "")
    LLM_MODEL_STRONG = os.getenv("LLM_MODEL_STRONG", "")
    LLM_TIMEOUT_MS = int(os.getenv("LLM_TIMEOUT_MS", "30000"))

    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "youmed_medical")

    # Model server
    MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8001")

    # Retrieval (Phase 1)
    DENSE_PREFETCH = int(os.getenv("DENSE_PREFETCH", "40"))
    SPARSE_PREFETCH = int(os.getenv("SPARSE_PREFETCH", "40"))
    FUSION_TOP_N = int(os.getenv("FUSION_TOP_N", "30"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "8"))
    MAX_CHUNKS_PER_ARTICLE = int(os.getenv("MAX_CHUNKS_PER_ARTICLE", "2"))
    FINAL_CONTEXT_K = int(os.getenv("FINAL_CONTEXT_K", "5"))


settings = Settings()
