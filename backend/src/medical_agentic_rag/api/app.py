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
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ingestion import embed_client, qdrant_store
from medical_agentic_rag.answer import format_sources
from medical_agentic_rag.api.schemas import (
    ChatRequest, ChatResponse, HealthResponse, Meta, ServiceHealth, SessionHistory, Source, Turn,
)
from medical_agentic_rag.config import settings
from medical_agentic_rag.graph.build import compile_graph
from medical_agentic_rag.graph.state import init_state
from medical_agentic_rag.observability import trace
from medical_agentic_rag.retrieval import article_index
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
@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.CHECKPOINT_DB).parent.mkdir(parents=True, exist_ok=True)  # deploy không có sẵn data/
    async with AsyncSqliteSaver.from_conn_string(settings.CHECKPOINT_DB) as saver:
        app.state.saver = saver
        app.state.graph = compile_graph(checkpointer=saver)
        try:  # dựng index tên bài ngay (khi deploy là đọc từ Qdrant), không để câu hỏi đầu tiên phải chờ
            n = await asyncio.to_thread(article_index.warm)
            print(f"[i] article index: {n} bài", flush=True)
        except Exception as exc:  # noqa: BLE001 — Qdrant chưa sẵn sàng: dựng lại ở lần gọi đầu tiên
            print(f"[!] chưa dựng được article index: {type(exc).__name__}: {exc}", flush=True)
        yield


app = FastAPI(title="Medical Agentic RAG API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


class RateLimiter:
    """Mỗi IP tối đa N câu/giờ + trần M câu/ngày (theo UTC) cho cả hệ thống. 0 = tắt giới hạn đó.

    Đếm trong bộ nhớ là đủ: deploy chỉ có 1 instance, và mục đích là không để ai tiêu hết credit LLM
    của link demo công khai — không phải hạn ngạch chính xác (restart thì đếm lại, chấp nhận được).
    Trần theo ngày mới là chốt chặn thật; giới hạn theo IP chỉ là best-effort vì X-Forwarded-For giả được.
    """

    def __init__(self, per_ip_hour: int, per_day: int):
        self.per_ip_hour, self.per_day = per_ip_hour, per_day
        self.hits: dict[str, deque] = defaultdict(deque)
        self.day, self.day_count = "", 0

    def check(self, ip: str, now: Optional[float] = None) -> Optional[str]:
        """Trả về lý do từ chối, hoặc None (và ghi nhận 1 lượt) nếu được hỏi."""
        now = time.time() if now is None else now
        today = time.strftime("%Y-%m-%d", time.gmtime(now))
        if today != self.day:
            self.day, self.day_count = today, 0
            self.hits.clear()
        if self.per_day and self.day_count >= self.per_day:
            return "Bản demo đã hết lượt hỏi của hôm nay. Bạn quay lại vào ngày mai nhé."
        q = self.hits[ip]
        while q and now - q[0] > 3600:
            q.popleft()
        if self.per_ip_hour and len(q) >= self.per_ip_hour:
            return f"Bạn đã hỏi {self.per_ip_hour} câu trong một giờ qua. Bạn thử lại sau ít phút nhé."
        q.append(now)
        self.day_count += 1
        return None


limiter = RateLimiter(settings.RATE_LIMIT_PER_IP_HOUR, settings.DAILY_QUESTION_CAP)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")  # sau proxy của Render
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def rate_limit(request: Request) -> None:
    reason = limiter.check(_client_ip(request))
    if reason:
        raise HTTPException(status_code=429, detail=reason)


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
@app.post("/api/v1/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit)])
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
        """Chạy graph với hai luồng sự kiện song song:

        "tasks"   — báo lúc node BẮT ĐẦU, để nhãn "Đang ..." hiện đúng bước đang chạy. Trước đây nhãn
                    lấy từ "updates" (phát khi node XONG), nên luôn chậm một bước: suốt lúc chấm bằng
                    chứng, UI vẫn hiện "Đang tìm trong 613 bài viết".
        "updates" — phần state mỗi node trả về, gom lại thành kết quả cuối.
        """
        config = {"configurable": {"thread_id": session_id, "on_token": on_token}}
        state = init_state(message, thread_id=session_id, trace_id=str(uuid.uuid4()))
        final: dict = dict(state)  # giữ trace_id / thread_id: "updates" chỉ chứa phần node trả về
        sent_sources = False
        last_label = None
        async for mode, chunk in app.state.graph.astream(
            state, config=config, stream_mode=["tasks", "updates"],
        ):
            if mode == "tasks":
                # sự kiện bắt đầu có "input", sự kiện kết thúc có "result"
                label = STEP_LABELS.get(chunk.get("name")) if "input" in chunk else None
                if label and label != last_label:  # retrieve_sub chạy song song nhiều nhánh cùng nhãn
                    last_label = label
                    await queue.put(("status", {"step": chunk["name"], "label": label}))
                continue
            for node, update in chunk.items():
                final.update(update or {})
                # gửi nguồn ngay khi đã có context, không chờ câu trả lời. Đánh số theo BÀI
                # (format_sources) cho khớp [n] trong câu trả lời — không theo từng đoạn.
                if not sent_sources and (update or {}).get("reranked_chunks"):
                    sent_sources = True
                    await queue.put(("sources", {"sources": format_sources(update["reranked_chunks"])}))
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


@app.post("/api/v1/chat/stream", dependencies=[Depends(rate_limit)])  # 429 trả về TRƯỚC khi mở stream
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
async def _probe_qdrant() -> ServiceHealth:
    """Đúng URL / key / collection mà retrieval dùng (ingestion.qdrant_store), không phải bản sao trong settings.
    Chỉ 200 mới là ổn: 401/403 là sai key, 404 là chưa có collection."""
    headers = {"api-key": qdrant_store.QDRANT_API_KEY} if qdrant_store.QDRANT_API_KEY else {}
    url = f"{qdrant_store.QDRANT_URL}/collections/{qdrant_store.QDRANT_COLLECTION}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=headers)
        return ServiceHealth(name="qdrant", ok=r.status_code == 200, detail=f"HTTP {r.status_code}")
    except Exception as exc:  # noqa: BLE001
        return ServiceHealth(name="qdrant", ok=False, detail=type(exc).__name__)


async def _probe_models() -> ServiceHealth:
    """Space ZeroGPU đang ngủ thì lời gọi này đánh thức nó (có thể mất cỡ một phút) — frontend hiện
    "đang khởi động" trong lúc chờ health trả lời. /ping của Space không tốn quota GPU."""
    where = f"Space {embed_client.MODEL_SPACE}" if embed_client.MODEL_SPACE else "model server local"
    try:
        await asyncio.wait_for(asyncio.to_thread(embed_client.ping), timeout=120)
        return ServiceHealth(name="model_server", ok=True, detail=where)
    except Exception as exc:  # noqa: BLE001
        return ServiceHealth(name="model_server", ok=False, detail=f"{where}: {type(exc).__name__}")


@app.get("/api/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    services = list(await asyncio.gather(_probe_qdrant(), _probe_models()))
    llm_ok = bool(settings.SHOPAIKEY_API_KEY and settings.LLM_MODEL_FAST)
    services.append(ServiceHealth(
        name="llm", ok=llm_ok, detail="đã cấu hình" if llm_ok else "thiếu SHOPAIKEY_API_KEY hoặc LLM_MODEL_*",
    ))
    return HealthResponse(ok=all(s.ok for s in services), services=services)
