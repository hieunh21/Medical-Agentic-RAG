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
    uvicorn.run(
        "medical_agentic_rag.api.app:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload="--reload" in sys.argv,
    )
