#!/usr/bin/env python3
"""CLI hỏi đáp end-to-end (Phase 1+2): graph retrieve -> rerank -> grade -> (sửa)* -> trả lời.

Usage (chạy từ repo root, cần model server :8001 và Qdrant đã ingest xong):
    python -m medical_agentic_rag.cli "Bệnh Addison là gì?"
"""
from __future__ import annotations

import asyncio
import sys
import uuid

from medical_agentic_rag.graph.build import get_graph
from medical_agentic_rag.graph.state import init_state


async def ask(question: str) -> dict:
    graph = get_graph()
    state = init_state(question, trace_id=str(uuid.uuid4()))
    result = await graph.ainvoke(state)
    return {
        "answer": result["final_answer"],
        "sources": result.get("sources") or [],
        "evidence_status": result.get("evidence_status"),
        "corrections": result.get("corrections", 0),
        "llm_calls": result.get("llm_calls", 0),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m medical_agentic_rag.cli \"câu hỏi\"", file=sys.stderr)
        return 1
    question = sys.argv[1]
    result = asyncio.run(ask(question))

    print("\n=== TRẢ LỜI ===")
    print(result["answer"])
    print("\n=== NGUỒN ===")
    for s in result["sources"]:
        print(f"[{s['n']}] {s['title']} — {s['url']}")
    print(f"\n(evidence_status={result['evidence_status']}, corrections={result['corrections']}, llm_calls={result['llm_calls']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
