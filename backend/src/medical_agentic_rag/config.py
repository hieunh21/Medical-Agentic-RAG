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
    LLM_MODEL_JUDGE = os.getenv("LLM_MODEL_JUDGE", "")
    # Tier rẻ cho task phân loại (grade_evidence chiếm ~34% token mỗi câu vì gửi cả context).
    # Để trống -> dùng LLM_MODEL_FAST, tức không đổi hành vi.
    LLM_MODEL_LITE = os.getenv("LLM_MODEL_LITE", "")
    LLM_TIMEOUT_MS = int(os.getenv("LLM_TIMEOUT_MS", "30000"))
    MAX_LLM_CALLS_PER_REQUEST = int(os.getenv("MAX_LLM_CALLS_PER_REQUEST", "10"))

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
    _rerank_confident = os.getenv("RERANK_CONFIDENT_SCORE", "").strip()
    RERANK_CONFIDENT_SCORE = float(_rerank_confident) if _rerank_confident else None

    # Corrective (Phase 2)
    MAX_CORRECTIONS = int(os.getenv("MAX_CORRECTIONS", "2"))
    COVERAGE_THRESHOLD = float(os.getenv("COVERAGE_THRESHOLD", "0.75"))
    # Số ký tự mỗi đoạn gửi cho grade_evidence. Grader chỉ cần phán "đoạn này có liên quan
    # không, trả lời khía cạnh nào" nên không cần nguyên văn; 0 = không cắt.
    GRADE_CHUNK_CHARS = int(os.getenv("GRADE_CHUNK_CHARS", "600"))

    # Hội thoại + Agent (Phase 3)
    CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "data/sessions.sqlite")
    HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "3"))
    AGENT_MAX_TOOL_CALLS = int(os.getenv("AGENT_MAX_TOOL_CALLS", "6"))
    AGENT_MAX_POOL = int(os.getenv("AGENT_MAX_POOL", "12"))
    AGENT_TIMEOUT_S = int(os.getenv("AGENT_TIMEOUT_S", "25"))
    # Giữ research_agent cho route "complex" sau 1 cờ: eval trên bộ complex (n=8,
    # mẫu nhỏ) cho thấy chưa thắng rõ baseline, nhưng khác với "comparison" (đã có
    # bằng chứng đủ mạnh để cắt), ở đây chưa đủ bằng chứng để bỏ — xem eval/run_routing_compare.py.
    AGENT_ENABLED = os.getenv("RESEARCH_AGENT_ENABLED", "true").lower() == "true"


settings = Settings()
