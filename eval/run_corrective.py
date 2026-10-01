#!/usr/bin/env python3
"""So sánh Phase 1 (retrieve->rerank, không sửa) với Phase 2 (graph, có sửa) — mục 7.5.

Rescue rate : trong các câu Phase 1 trượt Article@5, bao nhiêu câu Phase 2 cứu được.
Harm rate   : trong các câu Phase 1 đúng, bao nhiêu câu Phase 2 lại làm sai.
Chi phí     : số lời gọi LLM trung bình của Phase 2 (Phase 1 không gọi LLM để retrieve).

Usage:
    python -m eval.run_corrective --testset eval/testset/h_full.jsonl
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

from medical_agentic_rag.graph.build import get_graph
from medical_agentic_rag.graph.state import init_state
from medical_agentic_rag.retrieval.hybrid import retrieve
from medical_agentic_rag.retrieval.rerank import rerank_and_select


def article_hit(chunks: List[dict], gold_urls: List[str], k: int = 5) -> bool:
    seen = []
    for c in chunks:
        if c["url"] not in seen:
            seen.append(c["url"])
        if len(seen) >= k:
            break
    return any(u in gold_urls for u in seen)


async def run_one(row: dict) -> dict:
    question = row["question"]

    candidates = retrieve(question)
    phase1_chunks = rerank_and_select(question, candidates)
    phase1_hit = article_hit(phase1_chunks, row["gold_urls"])

    graph = get_graph()
    state = init_state(question, trace_id=str(uuid.uuid4()))
    result = await graph.ainvoke(state)
    phase2_hit = article_hit(result.get("reranked_chunks") or [], row["gold_urls"])

    return {
        "id": row["id"], "difficulty": row.get("difficulty"),
        "phase1_hit": phase1_hit, "phase2_hit": phase2_hit,
        "corrections": result.get("corrections", 0), "llm_calls": result.get("llm_calls", 0),
    }


async def main_async(testset_path: str, out_path: str, limit: int = 0, resume: bool = False) -> None:
    with open(testset_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip() and "gold_urls" in json.loads(line)]
    if limit > 0:
        rows = rows[:limit]

    results: List[dict] = []
    done_ids: set[str] = set()
    out_mode = "a"
    if resume and Path(out_path).exists():
        with open(out_path, encoding="utf-8") as f:
            results = [json.loads(line) for line in f if line.strip()]
        done_ids = {r["id"] for r in results}
        print(f"[i] resume: {len(done_ids)} câu đã có kết quả, bỏ qua", file=sys.stderr)
    else:
        out_mode = "w"

    with open(out_path, out_mode, encoding="utf-8") as out_f:
        for i, row in enumerate(rows, 1):
            if row["id"] in done_ids:
                continue
            r = await run_one(row)
            results.append(r)
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"[{i}/{len(rows)}] {row['id']} phase1={r['phase1_hit']} phase2={r['phase2_hit']} "
                  f"corrections={r['corrections']} llm_calls={r['llm_calls']}", file=sys.stderr)

    fails_p1 = [r for r in results if not r["phase1_hit"]]
    oks_p1 = [r for r in results if r["phase1_hit"]]
    rescued = [r for r in fails_p1 if r["phase2_hit"]]
    harmed = [r for r in oks_p1 if not r["phase2_hit"]]

    rescue_rate = len(rescued) / len(fails_p1) if fails_p1 else float("nan")
    harm_rate = len(harmed) / len(oks_p1) if oks_p1 else float("nan")
    avg_calls = sum(r["llm_calls"] for r in results) / len(results)
    avg_corrections = sum(r["corrections"] for r in results) / len(results)

    print(f"\nn={len(results)}  phase1_fails={len(fails_p1)}  phase1_oks={len(oks_p1)}")
    print(f"Rescue rate = {rescue_rate:.3f}  ({len(rescued)}/{len(fails_p1)})")
    print(f"Harm rate   = {harm_rate:.3f}  ({len(harmed)}/{len(oks_p1)})")
    print(f"Avg llm_calls (Phase2) = {avg_calls:.2f}")
    print(f"Avg corrections        = {avg_corrections:.2f}")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True)
    ap.add_argument("--out", default="data/eval_corrective_results.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    asyncio.run(main_async(args.testset, args.out, args.limit, args.resume))
    return 0


if __name__ == "__main__":
    sys.exit(main())
