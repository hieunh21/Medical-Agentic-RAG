#!/usr/bin/env python3
"""Chạy API backend từ repo root: python serve.py (mặc định http://localhost:8000)

Cần Qdrant (:6333) và model server (:8001) đang chạy — xem GET /api/v1/health.
Frontend dev server ở :5173 (xem frontend/), đã nằm trong CORS allow list.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

import uvicorn

if __name__ == "__main__":
    # Render (và đa số PaaS) cấp cổng qua biến PORT và cần nghe trên mọi interface.
    on_paas = bool(os.getenv("PORT"))
    uvicorn.run(
        "medical_agentic_rag.api.app:app",
        host=os.getenv("API_HOST") or ("0.0.0.0" if on_paas else "127.0.0.1"),
        port=int(os.getenv("PORT") or os.getenv("API_PORT", "8000")),
        reload="--reload" in sys.argv,
        proxy_headers=on_paas,  # tin X-Forwarded-* của proxy Render (IP thật cho rate limit, https)
        forwarded_allow_ips="*" if on_paas else None,
    )
