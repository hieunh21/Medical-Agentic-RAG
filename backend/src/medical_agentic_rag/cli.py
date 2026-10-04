#!/usr/bin/env python3
"""CLI hỏi đáp end-to-end (Phase 1+2+3): graph đầy đủ + checkpointer SQLite.

Usage (chạy từ repo root, cần model server :8001 và Qdrant đã ingest xong):
    python -m medical_agentic_rag.cli "Bệnh Addison là gì?"
    python -m medical_agentic_rag.cli "Còn triệu chứng thì sao?" my-thread-id   # lượt nối tiếp
"""
from __future__ import annotations

import asyncio
import sys
import time
import uuid

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from medical_agentic_rag.config import settings
from medical_agentic_rag.graph.build import compile_graph
from medical_agentic_rag.graph.state import init_state
from medical_agentic_rag.observability import trace


async def ask(question: str, thread_id: str) -> dict:
    async with AsyncSqliteSaver.from_conn_string(settings.CHECKPOINT_DB) as saver:
        graph = compile_graph(checkpointer=saver)
        state = init_state(question, thread_id=thread_id, trace_id=str(uuid.uuid4()))
        t0 = time.perf_counter()
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": thread_id}})
        trace.log_request(result, round((time.perf_counter() - t0) * 1000, 1))
    return {
        "answer": result["final_answer"],
        "sources": result.get("sources") or [],
        "evidence_status": result.get("evidence_status"),
        "question_type": result.get("question_type"),
        "standalone_question": result.get("standalone_question"),
        "corrections": result.get("corrections", 0),
        "agent_tool_calls": result.get("agent_tool_calls", 0),
        "llm_calls": result.get("llm_calls", 0),
        "safety_label": result.get("safety_label"),
        "claims": result.get("claims") or {},
        "thread_id": thread_id,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m medical_agentic_rag.cli \"câu hỏi\" [thread_id]", file=sys.stderr)
        return 1
    question = sys.argv[1]
    thread_id = sys.argv[2] if len(sys.argv) > 2 else str(uuid.uuid4())
    result = asyncio.run(ask(question, thread_id))

    print(f"\n=== CÂU HỎI ĐỘC LẬP === ({result['question_type']})")
    print(result["standalone_question"])
    print("\n=== TRẢ LỜI ===")
    print(result["answer"])
    print("\n=== NGUỒN ===")
    for s in result["sources"]:
        print(f"[{s['n']}] {s['title']} — {s['url']}")
    print(
        f"\n(evidence_status={result['evidence_status']}, corrections={result['corrections']}, "
        f"agent_tool_calls={result['agent_tool_calls']}, llm_calls={result['llm_calls']}, "
        f"thread_id={result['thread_id']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
