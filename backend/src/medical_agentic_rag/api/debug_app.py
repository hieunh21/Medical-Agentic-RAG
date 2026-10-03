"""FastAPI debug app: 1 trang HTML + 1 endpoint /ask để test graph bằng tay.

Không phải API sản phẩm (xem mục 16 trong thiết kế cho /api/v1/chat) — chỉ để
quan sát toàn bộ state (route, coverage, corrections, llm_calls, context chunks)
khi dev/test, thay vì đọc log CLI. Checkpointer mở 1 lần lúc khởi động app (lifespan)
để hội thoại nhiều lượt dùng chung 1 connection SQLite.

Usage: python debug_ui.py
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from medical_agentic_rag.config import settings
from medical_agentic_rag.graph.build import compile_graph
from medical_agentic_rag.graph.state import init_state

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSqliteSaver.from_conn_string(settings.CHECKPOINT_DB) as saver:
        app.state.graph = compile_graph(checkpointer=saver)
        yield


app = FastAPI(title="medical-agentic-rag debug UI", lifespan=lifespan)


class AskReq(BaseModel):
    question: str
    thread_id: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "debug.html")


@app.post("/ask")
async def ask(req: AskReq) -> dict:
    thread_id = req.thread_id or str(uuid.uuid4())
    state = init_state(req.question, thread_id=thread_id, trace_id=str(uuid.uuid4()))
    t0 = time.perf_counter()
    result = await app.state.graph.ainvoke(state, config={"configurable": {"thread_id": thread_id}})
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    chunks = result.get("reranked_chunks") or []
    return {
        "thread_id": thread_id,
        "question": req.question,
        "standalone_question": result.get("standalone_question"),
        "is_followup": result.get("is_followup", False),
        "question_type": result.get("question_type"),
        "answer": result.get("final_answer", ""),
        "sources": result.get("sources") or [],
        "evidence_status": result.get("evidence_status"),
        "coverage": result.get("coverage"),
        "n_relevant": result.get("n_relevant"),
        "missing_aspects": result.get("missing_aspects") or [],
        "corrections": result.get("corrections", 0),
        "agent_tool_calls": result.get("agent_tool_calls", 0),
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
