#!/usr/bin/env python3
"""A/B cho route section_lookup: lọc cứng theo 1 bài (cũ) vs gộp ứng viên lọc + không lọc (mới).

Mỗi câu hỏi H và R được analyze_query đúng 1 lần (cache ở data/eval_section_lookup_analyze.jsonl);
những câu bị xếp vào section_lookup chạy cả hai cách retrieve trên CÙNG entities / target_section_types,
nên khác biệt chỉ đến từ cách lấy ứng viên. Không gọi LLM ngoài bước analyze.

Lưu ý: các câu H-059, R-082, R-120 là nguyên nhân dẫn tới thay đổi này và nằm trong split test; số trên
split test vì vậy không còn "mù". Dùng split dev làm bằng chứng độc lập.

Usage:
    python -m eval.run_section_lookup_compare
    python -m eval.run_section_lookup_compare --limit 20      # chạy thử
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

from eval.run_ablation import fmt_rate  # noqa: E402
from medical_agentic_rag.graph.nodes.analyze import analyze_query  # noqa: E402
from medical_agentic_rag.graph.nodes.route_retrieve import _section_lookup_sync  # noqa: E402

TESTSET = Path("eval/testset")
CACHE = Path("data/eval_section_lookup_analyze.jsonl")
TIERS = ["easy", "medium", "hard"]


def load_questions() -> list[dict]:
    rows = []
    for name, group in (("h_full.jsonl", "H"), ("r_full.jsonl", "R")):
        for line in open(TESTSET / name, encoding="utf-8"):
            if line.strip():
                rows.append({**json.loads(line), "group": group})
    return rows


async def analyze_all(rows: list[dict], concurrency: int) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if CACHE.exists():
        for line in open(CACHE, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                cache[r["id"]] = r
    sem = asyncio.Semaphore(concurrency)
    todo = [r for r in rows if r["id"] not in cache]
    print(f"analyze: {len(todo)} câu cần chạy ({len(cache)} đã có cache)", flush=True)

    async def one(q: dict) -> None:
        async with sem:
            try:
                state = {"standalone_question": q["question"], "llm_calls": 0, "trace_id": str(uuid.uuid4())}
                out = await analyze_query(state)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {q['id']}: {type(exc).__name__}", file=sys.stderr)
                return
        row = {"id": q["id"], "question_type": out.get("question_type"), "entities": out.get("entities") or [],
               "target_section_types": out.get("target_section_types") or []}
        cache[q["id"]] = row
        with open(CACHE, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(q) for q in todo))
    return cache


def hit(chunks: list[dict], q: dict) -> tuple[bool, bool]:
    urls: list[str] = []
    for c in chunks:
        if c["url"] not in urls:
            urls.append(c["url"])
    article = any(u in q["gold_urls"] for u in urls[:5])
    chunk = q.get("gold_chunk_id") in [c["chunk_id"] for c in chunks] if q.get("gold_chunk_id") else False
    return article, chunk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="chỉ lấy N câu section_lookup đầu (chạy thử)")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    rows = load_questions()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    analysis = asyncio.run(analyze_all(rows, args.concurrency))
    targets = [q for q in rows if analysis.get(q["id"], {}).get("question_type") == "section_lookup"]
    if args.limit:
        targets = targets[: args.limit]
    print(f"{len(targets)}/{len(rows)} câu được xếp vào section_lookup\n", flush=True)

    res: list[dict] = []
    for i, q in enumerate(targets, 1):
        a = analysis[q["id"]]
        state = {"standalone_question": q["question"], "entities": a["entities"],
                 "target_section_types": a["target_section_types"]}
        old = hit(_section_lookup_sync(state, widen=False)[1], q)
        new = hit(_section_lookup_sync(state, widen=True)[1], q)
        res.append({"q": q, "old": old, "new": new})
        if i % 20 == 0:
            print(f"  {i}/{len(targets)}", flush=True)

    def block(label: str, rs: list[dict]) -> None:
        n = len(rs)
        if not n:
            return
        line = f"{label:22s} n={n:3d}  Article@5: cũ {fmt_rate(sum(r['old'][0] for r in rs), n)}  mới {fmt_rate(sum(r['new'][0] for r in rs), n)}"
        rr = [r for r in rs if r["q"].get("gold_chunk_id")]
        if rr:
            line += f"   chunk-hit: cũ {fmt_rate(sum(r['old'][1] for r in rr), len(rr))}  mới {fmt_rate(sum(r['new'][1] for r in rr), len(rr))}"
        print(line)

    print("=== Tổng ===")
    block("tất cả", res)
    for g in ("H", "R"):
        block(f"nhóm {g}", [r for r in res if r["q"]["group"] == g])
    print("\n=== Theo split ===")
    for sp in ("dev", "test"):
        block(f"split {sp}", [r for r in res if r["q"].get("split") == sp])
    print("\n=== Theo độ khó ===")
    for t in TIERS:
        block(f"mức {t}", [r for r in res if r["q"].get("difficulty") == t])

    gained = [r for r in res if r["new"][0] and not r["old"][0]]
    lost = [r for r in res if r["old"][0] and not r["new"][0]]
    print(f"\nCứu được {len(gained)} câu, làm mất {len(lost)} câu (Article@5)")
    for tag, rs in (("+", gained), ("-", lost)):
        for r in rs:
            print(f"  {tag} {r['q']['id']} [{r['q'].get('split')}/{r['q'].get('difficulty')}] {r['q']['question'][:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
