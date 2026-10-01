"""FastAPI debug app: 1 trang HTML + 1 endpoint /ask để test graph bằng tay.

Không phải API sản phẩm (xem mục 16 trong thiết kế cho /api/v1/chat) — chỉ để
quan sát toàn bộ state (coverage, corrections, llm_calls, context chunks) khi
dev/test, thay vì đọc log CLI.

Usage: python debug_ui.py
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from medical_agentic_rag.graph.build import get_graph
from medical_agentic_rag.graph.state import init_state

app = FastAPI(title="medical-agentic-rag debug UI")

STATIC_DIR = Path(__file__).parent / "static"


class AskReq(BaseModel):
    question: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "debug.html")


@app.post("/ask")
async def ask(req: AskReq) -> dict:
    graph = get_graph()
    state = init_state(req.question, trace_id=str(uuid.uuid4()))
    t0 = time.perf_counter()
    result = await graph.ainvoke(state)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    chunks = result.get("reranked_chunks") or []
    return {
        "question": req.question,
        "answer": result.get("final_answer", ""),
        "sources": result.get("sources") or [],
        "evidence_status": result.get("evidence_status"),
        "coverage": result.get("coverage"),
        "n_relevant": result.get("n_relevant"),
        "missing_aspects": result.get("missing_aspects") or [],
        "corrections": result.get("corrections", 0),
        "llm_calls": result.get("llm_calls", 0),
        "elapsed_ms": elapsed_ms,
        "context_chunks": [
            {
                "title": c.get("title"),
                "section_type": c.get("section_type"),
                "section_path": c.get("section_path"),
                "url": c.get("url"),
                "rerank_score": c.get("rerank_score"),
                "text": c.get("text", "")[:500],
            }
            for c in chunks
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
