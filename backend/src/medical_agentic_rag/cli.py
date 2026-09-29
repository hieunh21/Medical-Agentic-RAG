#!/usr/bin/env python3
"""CLI thử end-to-end baseline Phase 1: hybrid retrieve -> rerank -> generate_answer.

Usage (chạy từ repo root, cần model server :8001 và Qdrant đã ingest xong):
    python -m medical_agentic_rag.cli "Bệnh Addison là gì?"
"""
from __future__ import annotations

import asyncio
import sys

from medical_agentic_rag.answer import build_prompt, format_sources
from medical_agentic_rag.llm.client import generate_answer
from medical_agentic_rag.retrieval.hybrid import retrieve
from medical_agentic_rag.retrieval.rerank import rerank_and_select

NO_INFO = "YouMed hiện chưa có bài viết về vấn đề này. Bạn nên hỏi ý kiến bác sĩ để được tư vấn chính xác."


async def ask(question: str) -> dict:
    candidates = retrieve(question)
    chunks = rerank_and_select(question, candidates)
    if not chunks:
        return {"answer": NO_INFO, "sources": []}

    prompt = build_prompt(question, chunks)
    answer = await generate_answer(prompt)
    return {"answer": answer, "sources": format_sources(chunks)}


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
