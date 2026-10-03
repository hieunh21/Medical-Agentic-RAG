#!/usr/bin/env python3
"""Routing accuracy + confusion matrix cho analyze_query (mục 8.7).

Gộp 3 nguồn: H (expected_route), M (subtype -> comparison/multi_aspect), và một
danh sách nhỏ câu ngoài phạm vi (out_of_scope) viết tay — vì 2 bộ H/M không có
sẵn ví dụ out_of_scope. Route "complex" chưa có bộ test riêng (xem ghi chú ở
cuối output) nên không nằm trong confusion matrix này.

Usage:
    python -m eval.run_routing_accuracy
    python -m eval.run_routing_accuracy --resume   # bỏ qua câu đã có kết quả
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

from medical_agentic_rag.graph.nodes.analyze import analyze_query

OUT_OF_SCOPE_QUESTIONS = [
    "Giá vàng hôm nay bao nhiêu một lượng?",
    "Công thức tính diện tích hình tròn là gì?",
    "Đội tuyển Việt Nam đá vòng loại World Cup khi nào?",
    "Cách nấu phở bò ngon chuẩn vị Hà Nội?",
    "Lãi suất gửi tiết kiệm ngân hàng hiện nay là bao nhiêu?",
    "Làm sao để học tiếng Anh nhanh?",
]

ROUTES = ["definition", "section_lookup", "symptom_to_condition", "comparison", "multi_aspect", "out_of_scope"]
RESULTS_PATH = "data/eval_routing_accuracy_results.jsonl"


def load_rows() -> List[dict]:
    rows: List[dict] = []
    with open("eval/testset/h_full.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append({"id": r["id"], "question": r["question"], "expected": r["expected_route"]})
    with open("eval/testset/m_full.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append({"id": r["id"], "question": r["question"], "expected": r["subtype"]})
    for i, q in enumerate(OUT_OF_SCOPE_QUESTIONS, 1):
        rows.append({"id": f"OOS-{i:02d}", "question": q, "expected": "out_of_scope"})
    return rows


async def predict(question: str) -> str:
    state = {"standalone_question": question, "llm_calls": 0, "trace_id": str(uuid.uuid4())}
    result = await analyze_query(state)
    return result["question_type"]


def print_report(results: List[dict]) -> None:
    matrix: dict[str, dict[str, int]] = {a: {p: 0 for p in ROUTES + ["other"]} for a in ROUTES}
    correct = 0
    for r in results:
        actual, pred = r["expected"], r["predicted"]
        bucket = pred if pred in ROUTES else "other"
        matrix.setdefault(actual, {p: 0 for p in ROUTES + ["other"]})
        matrix[actual][bucket] += 1
        correct += pred == actual

    print(f"\n=== Routing accuracy: {correct}/{len(results)} = {correct/len(results):.3f} ===\n")
    header = ["actual\\pred"] + ROUTES + ["other"]
    print(" | ".join(f"{h:12s}" for h in header))
    for actual in ROUTES:
        row_counts = matrix.get(actual, {})
        total = sum(row_counts.values())
        if total == 0:
            continue
        cells = [f"{row_counts.get(p, 0):12d}" for p in ROUTES + ["other"]]
        print(" | ".join([f"{actual:12s}"] + cells) + f"   (n={total})")

    print("\n=== Precision/Recall theo route ===")
    for route in ROUTES:
        tp = matrix.get(route, {}).get(route, 0)
        support = sum(matrix.get(route, {}).values())
        predicted_as = sum(matrix.get(a, {}).get(route, 0) for a in ROUTES)
        recall = tp / support if support else float("nan")
        precision = tp / predicted_as if predicted_as else float("nan")
        print(f"{route:22s} precision={precision:.3f}  recall={recall:.3f}  support={support}")

    print("\n(Chưa có route 'complex' trong confusion matrix này — cần bộ test complex riêng.)")


async def main_async(resume: bool) -> None:
    rows = load_rows()
    results: List[dict] = []
    done_ids: set[str] = set()
    out_mode = "a"
    if resume and Path(RESULTS_PATH).exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            results = [json.loads(line) for line in f if line.strip()]
        done_ids = {r["id"] for r in results}
        print(f"[i] resume: {len(done_ids)} câu đã có kết quả, bỏ qua", file=sys.stderr)
    else:
        out_mode = "w"

    with open(RESULTS_PATH, out_mode, encoding="utf-8") as out_f:
        for i, row in enumerate(rows, 1):
            if row["id"] in done_ids:
                continue
            pred = await predict(row["question"])
            r = {"id": row["id"], "expected": row["expected"], "predicted": pred}
            results.append(r)
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()
            ok = pred == row["expected"]
            print(f"[{i}/{len(rows)}] expected={row['expected']:22s} predicted={pred:22s} {'OK' if ok else 'MISS'}",
                  file=sys.stderr)

    print_report(results)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    asyncio.run(main_async(args.resume))
    return 0


if __name__ == "__main__":
    sys.exit(main())
