"""Trace JSONL (mục 10.3): mỗi lời gọi LLM, mỗi node và mỗi request một dòng.

Phân biệt bằng trường "kind": "llm" | "node" | "request". Mọi dòng có trace_id.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

TRACE_LOG = os.getenv("TRACE_LOG", "data/traces.jsonl")


def _write(row: dict) -> None:
    try:
        path = Path(TRACE_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass  # trace hỏng không được làm hỏng request


def log_llm(
    state: dict, task: str, attempt: int, t0: float, ok: bool, error: str | None = None,
    tokens_in: Optional[int] = None, tokens_out: Optional[int] = None, detail: Optional[str] = None,
) -> None:
    _write({
        "kind": "llm", "ts": time.time(), "trace_id": state.get("trace_id"),
        "task": task, "attempt": attempt,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "ok": ok, "error": error, "detail": detail, "tokens_in": tokens_in, "tokens_out": tokens_out,
    })


def traced(name: str, fn: Callable) -> Callable:
    """Bọc node LangGraph (sync hoặc async): ghi thời gian + vài trường quyết định nhẹ.

    Giữ nguyên chữ ký của node: node nhận (state, config) thì wrapper cũng nhận config và
    truyền xuống — LangGraph quyết định có truyền config dựa trên chữ ký này.
    """
    def _log(state: Any, t0: float, result: Any) -> None:
        st = state if isinstance(state, dict) else {}
        out = result if isinstance(result, dict) else {}
        row = {
            "kind": "node", "ts": time.time(), "trace_id": st.get("trace_id"), "node": name,
            "t_ms": round((time.perf_counter() - t0) * 1000, 1),
            "llm_calls": out.get("llm_calls", st.get("llm_calls")),
        }
        for key in ("coverage", "n_relevant", "question_type", "safety_label", "evidence_status", "regenerate"):
            if key in out:
                row[key] = out[key]
        _write(row)

    takes_config = "config" in inspect.signature(fn).parameters

    if asyncio.iscoroutinefunction(fn):
        if takes_config:
            @functools.wraps(fn)
            async def awrapper_cfg(state, config):
                t0 = time.perf_counter()
                result = await fn(state, config)
                _log(state, t0, result)
                return result
            return awrapper_cfg

        @functools.wraps(fn)
        async def awrapper(state):
            t0 = time.perf_counter()
            result = await fn(state)
            _log(state, t0, result)
            return result
        return awrapper

    @functools.wraps(fn)
    def wrapper(state):
        t0 = time.perf_counter()
        result = fn(state)
        _log(state, t0, result)
        return result
    return wrapper


def log_request(result: dict, latency_ms: float) -> None:
    """Một dòng tổng kết cho cả request (gọi từ cli.py / api sau khi graph chạy xong)."""
    from langchain_core.messages import HumanMessage
    from medical_agentic_rag.graph.routing import route_by_type

    stopped = result.get("safety_label") in ("emergency", "self_harm")
    _write({
        "kind": "request", "ts": time.time(),
        "trace_id": result.get("trace_id"), "thread_id": result.get("thread_id"),
        "turn": sum(isinstance(m, HumanMessage) for m in result.get("messages") or []),
        "is_followup": result.get("is_followup", False),
        "question_type": result.get("question_type"),
        "route": "safety_response" if stopped else route_by_type(result),
        "safety_label": result.get("safety_label"),
        "corrections": result.get("corrections", 0),
        "agent_tool_calls": result.get("agent_tool_calls", 0),
        "coverage_trace": result.get("coverage_trace") or [],
        "llm_calls": result.get("llm_calls", 0),
        "regenerations": result.get("regenerations", 0),
        "claims": result.get("claims") or {},
        "evidence_status": result.get("evidence_status"),
        "latency_ms": latency_ms,
    })
