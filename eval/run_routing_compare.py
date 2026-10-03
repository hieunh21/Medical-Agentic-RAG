#!/usr/bin/env python3
"""So sánh 3 cách xử lý câu hỏi nhóm M (mục 8.7): không routing / subquery song song /
research agent. Gọi trực tiếp từng node (không qua analyze_query) để đảm bảo cả 3 cách
đều chạy trên đúng cùng một câu hỏi, không phụ thuộc việc analyze_query phân loại đúng hay không.

Metric:
  article_coverage  = tỉ lệ gold_urls có mặt trong context cuối cùng
  section_coverage  = (chỉ multi_aspect) tỉ lệ expected_section_types có mặt trong các
                       chunk thuộc đúng gold article

Usage:
    python -m eval.run_routing_compare --testset eval/testset/m_full.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

from medical_agentic_rag.agent.loop import research_agent
from medical_agentic_rag.graph.nodes.subquery import fan_out, merge_evidence, plan_subqueries, retrieve_sub
from medical_agentic_rag.retrieval.hybrid import retrieve
from medical_agentic_rag.retrieval.rerank import rerank_and_select

METHODS = ["no_routing", "subquery", "agent"]


def fresh_state(question: str) -> dict:
    return {
        "question": question, "standalone_question": question,
        "entities": [], "aspects": [], "llm_calls": 0, "corrections": 0,
        "retrieved_pool": {}, "trace_id": str(uuid.uuid4()),
    }


async def run_no_routing(question: str) -> List[dict]:
    candidates = retrieve(question)
    return rerank_and_select(question, candidates)


async def run_subquery(question: str) -> List[dict]:
    state = fresh_state(question)
    state.update(await plan_subqueries(state))
    sub_results: List[dict] = []
    for send in fan_out(state):
        sub_results.extend((await retrieve_sub(send.arg))["sub_results"])
    state["sub_results"] = sub_results
    merged = await merge_evidence(state)
    return merged["reranked_chunks"]


async def run_agent(question: str) -> List[dict]:
    state = fresh_state(question)
    result = await research_agent(state)
    chunks = list(result["retrieved_pool"].values())
    return rerank_and_select(question, chunks)


RUNNERS = {"no_routing": run_no_routing, "subquery": run_subquery, "agent": run_agent}


def article_coverage(chunks: List[dict], gold_urls: List[str]) -> float:
    if not gold_urls:
        return float("nan")
    got = {c["url"] for c in chunks}
    return len(got & set(gold_urls)) / len(gold_urls)


def section_coverage(chunks: List[dict], gold_urls: List[str], expected: List[str]) -> float:
    if not expected:
        return float("nan")
    got_types = {c["section_type"] for c in chunks if c["url"] in gold_urls}
    return len(got_types & set(expected)) / len(expected)


async def main_async(testset_path: str) -> None:
    with open(testset_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    by_method: dict[str, list[dict]] = {m: [] for m in METHODS}

    for i, row in enumerate(rows, 1):
        for method in METHODS:
            chunks = await RUNNERS[method](row["question"])
            ac = article_coverage(chunks, row["gold_urls"])
            sc = section_coverage(chunks, row["gold_urls"], row.get("expected_section_types") or [])
            by_method[method].append({
                "id": row["id"], "difficulty": row.get("difficulty"), "subtype": row.get("subtype"),
                "article_coverage": ac, "section_coverage": sc,
            })
            print(f"[{i}/{len(rows)}] {row['id']} ({row.get('subtype')},{row.get('difficulty')}) "
                  f"{method}: article={ac:.2f} section={sc if sc==sc else '-'}", file=sys.stderr)

    def avg(xs: List[float]) -> float:
        xs = [x for x in xs if x == x]  # bỏ nan
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\n=== Tổng kết (n={len(rows)}) ===")
    for method in METHODS:
        rs = by_method[method]
        ac = avg([r["article_coverage"] for r in rs])
        sc = avg([r["section_coverage"] for r in rs])
        print(f"{method:12s}  article_coverage={ac:.3f}  section_coverage={sc:.3f}")
        for diff in ("easy", "medium", "hard"):
            sub = [r for r in rs if r["difficulty"] == diff]
            if sub:
                print(f"    {diff:7s} (n={len(sub)})  article={avg([r['article_coverage'] for r in sub]):.3f}  "
                      f"section={avg([r['section_coverage'] for r in sub]):.3f}")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True)
    args = ap.parse_args(argv)
    asyncio.run(main_async(args.testset))
    return 0


if __name__ == "__main__":
    sys.exit(main())
