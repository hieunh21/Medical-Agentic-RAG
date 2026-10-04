"""FastAPI app cho frontend chat (mục 16).

Checkpointer mở một lần trong lifespan vì AsyncSqliteSaver là async context manager cần giữ
mở suốt vòng đời app; graph compile một lần dùng chung cho mọi request.

Streaming (POST /api/v1/chat/stream, SSE):
    status  — tên bước đang chạy, để UI không có cảm giác treo (P95 pipeline ~30-50s)
    sources — danh sách nguồn, gửi ngay khi rerank xong, trước khi câu trả lời có
    token   — từng mảnh text của bản nháp
    done    — câu trả lời CHỐT (sau citation_validator) + meta; UI thay toàn bộ text bằng cái này
    error   — lỗi không lường được

Vì citation_validator chạy SAU generate, bản nháp đã stream có thể bị lọc câu hoặc viết lại.
Hợp đồng ở đây: `token` là tạm, `done` mới là chốt — UI phải thay text khi nhận done.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from medical_agentic_rag.api.schemas import (
    ChatRequest, ChatResponse, HealthResponse, Meta, ServiceHealth, SessionHistory, Source, Turn,
)
from medical_agentic_rag.config import settings
from medical_agentic_rag.graph.build import compile_graph
from medical_agentic_rag.graph.state import init_state
from medical_agentic_rag.observability import trace
from medical_agentic_rag.retrieval.article_index import get_titles

# Tên bước hiển thị cho người dùng — không lộ tên node nội bộ.
STEP_LABELS = {
    "safety_rules": "Đang kiểm tra an toàn",
    "condense_question": "Đang đọc lại ngữ cảnh hội thoại",
    "analyze_query": "Đang phân tích câu hỏi",
    "retrieve_definition": "Đang tìm trong 613 bài viết",
    "retrieve_section_lookup": "Đang tìm trong 613 bài viết",
    "hybrid_retrieve": "Đang tìm trong 613 bài viết",
    "plan_subqueries": "Đang tách câu hỏi thành nhiều phần",
    "retrieve_sub": "Đang tìm cho từng phần",
    "merge_evidence": "Đang gộp kết quả",
    "research_agent": "Đang tự tra cứu nhiều bài",
    "rerank": "Đang chọn đoạn liên quan nhất",
    "grade_evidence": "Đang đánh giá bằng chứng",
    "targeted_retrieve": "Đang tìm thêm cho phần còn thiếu",
    "rewrite_query": "Đang thử cách tìm khác",
    "generate_answer": "Đang viết câu trả lời",
    "citation_validator": "Đang kiểm tra trích dẫn",
}
ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSqliteSaver.from_conn_string(settings.CHECKPOINT_DB) as saver:
        app.state.saver = saver
        app.state.graph = compile_graph(checkpointer=saver)
        yield


app = FastAPI(title="Medical Agentic RAG API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


# --------------------------------------------------------------------------- helper
def _sources(result: dict) -> list[Source]:
    return [Source(**s) for s in (result.get("sources") or [])]


def _meta(result: dict, elapsed_ms: float) -> Meta:
    chunks = result.get("reranked_chunks") or []
    return Meta(
        question_type=result.get("question_type"),
        standalone_question=result.get("standalone_question"),
        is_followup=result.get("is_followup", False),
        safety_label=result.get("safety_label"),
        evidence_status=result.get("evidence_status"),
        coverage=result.get("coverage"),
        n_relevant=result.get("n_relevant"),
        n_chunks=len(chunks),
        n_articles=len({c["article_id"] for c in chunks}),
        corrections=result.get("corrections", 0),
        regenerations=result.get("regenerations", 0),
        agent_tool_calls=result.get("agent_tool_calls", 0),
        llm_calls=result.get("llm_calls", 0),
        claims=result.get("claims") or {},
        elapsed_ms=elapsed_ms,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# --------------------------------------------------------------------------- chat
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Không stream — tiện cho test bằng curl và cho client không dùng SSE."""
    session_id = req.session_id or str(uuid.uuid4())
    t0 = time.perf_counter()
    result = await app.state.graph.ainvoke(
        init_state(req.message, thread_id=session_id, trace_id=str(uuid.uuid4())),
        config={"configurable": {"thread_id": session_id}},
    )
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    trace.log_request(result, elapsed)
    return ChatResponse(
        session_id=session_id, answer=result.get("final_answer", ""),
        sources=_sources(result), meta=_meta(result, elapsed),
    )


async def _chat_events(message: str, session_id: str) -> AsyncIterator[str]:
    queue: asyncio.Queue = asyncio.Queue()
    t0 = time.perf_counter()

    async def on_token(piece: str) -> None:
        await queue.put(("token", {"text": piece}))

    async def drive() -> dict:
        """Chạy graph, đẩy status/sources vào queue theo từng node xong."""
        final: dict = {}
        config = {"configurable": {"thread_id": session_id, "on_token": on_token}}
        state = init_state(message, thread_id=session_id, trace_id=str(uuid.uuid4()))
        sent_sources = False
        async for chunk in app.state.graph.astream(state, config=config, stream_mode="updates"):
            for node, update in chunk.items():
                final.update(update or {})
                if node in STEP_LABELS:
                    await queue.put(("status", {"step": node, "label": STEP_LABELS[node]}))
                # gửi nguồn ngay khi đã có context, không chờ câu trả lời
                if not sent_sources and (update or {}).get("reranked_chunks"):
                    chunks = update["reranked_chunks"]
                    sent_sources = True
                    await queue.put(("sources", {"sources": [
                        {"n": i, "title": c["title"], "url": c["url"], "updated_date": c.get("updated_date")}
                        for i, c in enumerate(chunks, 1)
                    ]}))
        return final

    task = asyncio.create_task(drive())
    try:
        while True:
            get = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({get, task}, return_when=asyncio.FIRST_COMPLETED)
            if get in done:
                event, data = get.result()
                yield _sse(event, data)
                continue
            get.cancel()  # graph xong: xả hết event còn trong queue rồi mới gửi done
            while not queue.empty():
                event, data = queue.get_nowait()
                yield _sse(event, data)
            break

        result = task.result()
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        trace.log_request(result, elapsed)
        # `sources` lúc này là bản chốt (validator có thể đã trả về rỗng khi lọc hết câu)
        yield _sse("done", {
            "session_id": session_id,
            "answer": result.get("final_answer", ""),
            "sources": [s.model_dump() for s in _sources(result)],
            "meta": _meta(result, elapsed).model_dump(),
        })
    except asyncio.CancelledError:  # client đóng tab
        task.cancel()
        raise
    except Exception as exc:  # noqa: BLE001 — lỗi nào cũng phải tới được frontend
        yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"[:300]})
    finally:
        if not task.done():
            task.cancel()


@app.post("/api/v1/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    session_id = req.session_id or str(uuid.uuid4())
    return StreamingResponse(
        _chat_events(req.message, session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Session-Id": session_id},
    )


# --------------------------------------------------------------------------- session
async def _load_state(session_id: str) -> Optional[dict]:
    snapshot = await app.state.graph.aget_state({"configurable": {"thread_id": session_id}})
    return snapshot.values if snapshot and snapshot.values else None


@app.get("/api/v1/sessions/{session_id}", response_model=SessionHistory)
async def get_session(session_id: str) -> SessionHistory:
    values = await _load_state(session_id)
    if values is None:
        raise HTTPException(status_code=404, detail="session không tồn tại")
    turns = [
        Turn(role="user" if isinstance(m, HumanMessage) else "assistant", content=str(m.content))
        for m in (values.get("messages") or [])
        if isinstance(m, (HumanMessage, AIMessage))
    ]
    return SessionHistory(
        session_id=session_id, turns=turns,
        active_articles=get_titles(values.get("active_article_ids") or []),
    )


@app.delete("/api/v1/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    await app.state.saver.adelete_thread(session_id)


# --------------------------------------------------------------------------- health
async def _probe(name: str, url: str) -> ServiceHealth:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(url)
        return ServiceHealth(name=name, ok=r.status_code < 500, detail=f"HTTP {r.status_code}")
    except Exception as exc:  # noqa: BLE001
        return ServiceHealth(name=name, ok=False, detail=type(exc).__name__)


@app.get("/api/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    services = list(await asyncio.gather(
        _probe("qdrant", f"{settings.QDRANT_URL}/collections/{settings.QDRANT_COLLECTION}"),
        _probe("model_server", f"{settings.MODEL_SERVER_URL}/docs"),
    ))
    services.append(ServiceHealth(
        name="llm", ok=bool(settings.SHOPAIKEY_API_KEY and settings.LLM_MODEL_FAST),
        detail="thiếu SHOPAIKEY_API_KEY hoặc LLM_MODEL_*" if not settings.SHOPAIKEY_API_KEY else "đã cấu hình",
    ))
    return HealthResponse(ok=all(s.ok for s in services), services=services)
