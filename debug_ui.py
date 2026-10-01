#!/usr/bin/env python3
"""Debug UI chạy từ repo root: python debug_ui.py (mặc định http://localhost:8002)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("medical_agentic_rag.api.debug_app:app", host="0.0.0.0", port=8002)
