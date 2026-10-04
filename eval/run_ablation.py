#!/usr/bin/env python3
"""Bảng ablation (mục 10.4): mỗi hàng bật thêm một thành phần, chạy trên cùng tập câu hỏi.

Hàng (tích luỹ): P1 baseline -> +P2 corrective -> +P3 routing -> +P3 research agent -> +P5 safety
-> +P5 citation validator. (Phase 4 đã bỏ.)

Câu hỏi (mặc định split=test):
  H  văn nói, viết tay      -> Article@5, must_mention, faithfulness, citation precision, từ chối nhầm
  R  sinh từ chunk, phân tầng -> Article@5, chunk-hit (mẫu --r-per-tier câu mỗi mức)
  O  ngoài phạm vi / không có thông tin -> no-info precision
  S  safety (chỉ chạy các hàng có safety) -> stop recall, tỉ lệ dừng nhầm, độ chính xác nhãn

Metric so với gold hoàn toàn khách quan (Article@5, chunk-hit, must_mention, nhãn safety, từ chối).
Faithfulness / citation precision dùng LLM-judge (task eval_judge) — CHƯA hiệu chuẩn bằng người, là chỉ
số tham khảo. Mặc định judge dùng cùng model với bên sinh câu trả lời (xem LLM_MODEL_*); đặt
EVAL_JUDGE_MODEL để dùng model khác. Mọi tỉ lệ kèm khoảng tin cậy Wilson 95%: n ở đây nhỏ.

Kết quả từng câu được cache ở data/ablation/<hàng>.jsonl (chạy lại sẽ tiếp tục, bỏ qua câu đã xong).

Usage (cần Qdrant, model server :8001, LLM key):
    python -m eval.run_ablation                      # toàn bộ hàng, split test
    python -m eval.run_ablation --split dev --configs P1,P2 --limit 10   # chạy thử
    python -m eval.run_ablation --report-only        # chỉ dựng lại bảng từ cache
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

from pydantic import BaseModel  # noqa: E402

from medical_agentic_rag.citation import validator as v  # noqa: E402
from medical_agentic_rag.graph.build import Features, compile_graph  # noqa: E402
from medical_agentic_rag.graph.nodes.validate import FILTERED_NOTE  # noqa: E402
from medical_agentic_rag.graph.state import init_state  # noqa: E402
from medical_agentic_rag.llm import client as llm_client  # noqa: E402
from medical_agentic_rag.llm.client import run_task  # noqa: E402
from medical_agentic_rag.llm.prompts import build_verify_claims_prompt  # noqa: E402
from medical_agentic_rag.llm.schemas import ClaimVerdicts  # noqa: E402
from medical_agentic_rag.llm.tasks import TASKS, TaskSpec  # noqa: E402
from medical_agentic_rag.safety.rules import STOP_LABELS, strip_accents  # noqa: E402
from medical_agentic_rag.safety.templates import EMERGENCY_TOPIC_NOTE, HIGH_RISK_PREFIX  # noqa: E402

TESTSET = Path("eval/testset")
OUT_DIR = Path("data/ablation")
TIERS = ["easy", "medium", "hard"]

# (key, nhãn bảng, Features). Tích luỹ từ trên xuống.
CONFIGS: list[tuple[str, str, Features]] = [
    ("P1", "P1: hybrid + rerank + generate",
     Features(routing=False, corrective=False, agent=False, safety=False, validator=False)),
    ("P2", "+ P2: corrective",
     Features(routing=False, corrective=True, agent=False, safety=False, validator=False)),
    ("P3", "+ P3: hội thoại + routing",
     Features(routing=True, corrective=True, agent=False, safety=False, validator=False)),
    ("P3A", "+ P3: research agent",
     Features(routing=True, corrective=True, agent=True, safety=False, validator=False)),
    ("P5S", "+ P5: safety",
     Features(routing=True, corrective=True, agent=True, safety=True, validator=False)),
    ("P5V", "+ P5: citation validator",
     Features(routing=True, corrective=True, agent=True, safety=True, validator=True)),
]

REFUSAL_RE = re.compile(r"chưa có (?:thông tin|bài)|ngoài phạm vi|không có thông tin|không đủ thông tin", re.IGNORECASE)

CONFIGS_BY_KEY = {k: f for k, _, f in CONFIGS}

TASKS["eval_judge"] = TaskSpec("eval_judge", 0.0, 2048)
llm_client.MODEL_BY_TIER["eval_judge"] = os.getenv("EVAL_JUDGE_MODEL") or llm_client.MODEL_BY_TIER["judge"]


# ----------------------------------------------------------------------------- dữ liệu
def _read(name: str) -> list[dict]:
    path = TESTSET / name
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def select_questions(split: str, r_per_tier: int, limit: Optional[int]) -> list[dict]:
    def in_split(r: dict) -> bool:
        return split == "all" or r.get("split") == split

    rows: list[dict] = []
    for r in _read("h_full.jsonl"):
        if in_split(r):
            rows.append({**r, "group": "H"})
    r_rows = [r for r in _read("r_full.jsonl") if in_split(r)]
    for tier in TIERS:
        pool = sorted((r for r in r_rows if r["difficulty"] == tier), key=lambda r: r["id"])
        step = max(1, len(pool) // max(r_per_tier, 1))
        rows.extend(pool[::step][:r_per_tier])
    rows.extend(r for r in _read("o_full.jsonl") if in_split(r))
    rows.extend(r for r in _read("s_full.jsonl") if in_split(r))
    if limit:  # chạy thử: giữ limit câu đầu mỗi nhóm
        seen: dict[str, int] = defaultdict(int)
        kept = []
        for r in rows:
            seen[r["group"]] += 1
            if seen[r["group"]] <= limit:
                kept.append(r)
        rows = kept
    return rows


# ----------------------------------------------------------------------------- chạy 1 câu
async def judge_answer(answer: str, chunks: list[dict]) -> Optional[dict]:
    """Faithfulness + citation precision bằng LLM-judge, dùng lại tách câu / bước 1 của validator."""
    # câu template do code chèn (khuyến cáo high-risk, ghi chú lược bỏ) không phải khẳng định cần nguồn
    for template in (HIGH_RISK_PREFIX, EMERGENCY_TOPIC_NOTE, FILTERED_NOTE):
        answer = answer.replace(template.strip(), "")
    answer = answer.strip()
    if not answer or REFUSAL_RE.search(answer) and "[" not in answer:
        return None
    sentences = v.split_sentences(answer)
    v.check_citations(sentences, len(chunks))
    claims = [s for s in sentences if s.needs_cite]
    if not claims:
        return None
    targets = v.to_verify(sentences)
    if targets:
        prompt = build_verify_claims_prompt([(s.id, s.text, s.refs) for s in targets], chunks)
        try:
            res: ClaimVerdicts = await run_task(
                "eval_judge", prompt, {"llm_calls": 0, "trace_id": "eval-judge"}, schema=ClaimVerdicts)
            v.apply_verdicts(sentences, {c.sentence_id: (c.verdict, c.reason) for c in res.verdicts})
        except Exception:  # noqa: BLE001 — judge lỗi: bỏ qua câu này, không làm hỏng cả lượt chạy
            return {"judge_error": True}
    counts = v.claim_counts(sentences)
    cited = [s for s in claims if s.refs]
    prec = (sum(1.0 for s in cited if s.verdict == "supported") + sum(0.5 for s in cited if s.verdict == "partial"))
    return {
        "n_claims": len(claims), "unsupported": sum(s.verdict == "unsupported" for s in claims),
        "n_cited": len(cited), "n_uncited": len(claims) - len(cited), "cite_score": prec, "counts": counts,
    }


async def run_one(graph, q: dict, with_judge: bool) -> dict:
    row = {"id": q["id"], "group": q["group"], "difficulty": q.get("difficulty")}
    t0 = time.perf_counter()
    try:
        state = init_state(q["question"], trace_id=str(uuid.uuid4()))
        result = await graph.ainvoke(state)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return row
    row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    chunks = result.get("reranked_chunks") or []
    answer = result.get("final_answer", "")
    urls: list[str] = []
    for c in chunks:
        if c["url"] not in urls:
            urls.append(c["url"])
    row.update(
        answer=answer, urls=urls[:5], chunk_ids=[c["chunk_id"] for c in chunks],
        llm_calls=result.get("llm_calls", 0), safety_label=result.get("safety_label"),
        evidence_status=result.get("evidence_status"), question_type=result.get("question_type"),
        refused=bool(REFUSAL_RE.search(answer)) and not re.search(r"\[\d+\]", answer),
    )
    if q["group"] == "H" and with_judge:
        row["judge"] = await judge_answer(answer, chunks)
    return row


async def run_config(
    key: str, features: Features, questions: list[dict], concurrency: int, with_judge: bool, suffix: str = "",
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{key}{suffix}.jsonl"
    done = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    done[r["id"]] = r
    todo = [q for q in questions if q["id"] not in done and (q["group"] != "S" or features.safety)]
    print(f"[{key}] {len(todo)} câu cần chạy ({len(done)} đã có cache)", flush=True)
    if not todo:
        return
    graph = compile_graph(features=features)
    sem = asyncio.Semaphore(concurrency)
    finished = 0

    async def worker(q: dict) -> None:
        nonlocal finished
        async with sem:
            row = await run_one(graph, q, with_judge)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        finished += 1
        if finished % 10 == 0 or finished == len(todo):
            print(f"[{key}] {finished}/{len(todo)}", flush=True)

    await asyncio.gather(*(worker(q) for q in todo))


# ----------------------------------------------------------------------------- metric
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_rate(k: float, n: int) -> str:
    if n == 0:
        return "—"
    lo, hi = wilson(round(k), n)
    return f"{k / n:.2f} [{lo:.2f}–{hi:.2f}]"


def p95(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1)]


def norm(s: str) -> str:
    return strip_accents(s)


def load_rows(key: str, suffix: str = "") -> dict[str, dict]:
    path = OUT_DIR / f"{key}{suffix}.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def gold_index(questions: list[dict]) -> dict[str, dict]:
    return {q["id"]: q for q in questions}


def metrics_for(rows: dict[str, dict], gold: dict[str, dict]) -> dict:
    """Mỗi metric là (số đạt, n) để in kèm khoảng tin cậy; vài metric là số thực."""
    m: dict = defaultdict(lambda: [0.0, 0])
    tier: dict = defaultdict(lambda: [0.0, 0])
    calls, lat = [], []
    n_err = 0
    for rid, r in rows.items():
        q = gold.get(rid)
        if q is None:
            continue
        if "error" in r:
            n_err += 1
            continue
        g = r["group"]
        calls.append(r["llm_calls"])
        lat.append(r["latency_ms"] / 1000)
        if g in ("H", "R"):
            hit = any(u in q["gold_urls"] for u in r["urls"])
            m[f"{g}_article5"][0] += hit
            m[f"{g}_article5"][1] += 1
            tier[(g, q["difficulty"])][0] += hit
            tier[(g, q["difficulty"])][1] += 1
            if g == "R":
                m["R_chunk"][0] += q["gold_chunk_id"] in r["chunk_ids"]
                m["R_chunk"][1] += 1
        if g == "H":
            ans = norm(r["answer"])
            must = q.get("must_mention") or []
            if must:
                m["H_must"][0] += sum(norm(x) in ans for x in must) / len(must)
                m["H_must"][1] += 1
            m["H_false_refusal"][0] += r["refused"]
            m["H_false_refusal"][1] += 1
            j = r.get("judge")
            if j and "n_claims" in j:
                m["H_faith"][0] += j["n_claims"] - j["unsupported"]
                m["H_faith"][1] += j["n_claims"]
                m["H_cov"][0] += j["n_cited"]
                m["H_cov"][1] += j["n_claims"]
                if j["n_cited"]:
                    m["H_cite"][0] += j["cite_score"]
                    m["H_cite"][1] += j["n_cited"]
        if g == "O":
            beh = q["expected_behavior"]
            ok = r["refused"] or (beh in ("decline_out_of_scope", "either") and r.get("question_type") == "out_of_scope")
            m["O_noinfo"][0] += ok
            m["O_noinfo"][1] += 1
            tier[("O", q["difficulty"])][0] += ok
            tier[("O", q["difficulty"])][1] += 1
        if g == "S":
            stop_pred = r["safety_label"] in STOP_LABELS
            if q["expected_action"] == "stop":
                m["S_stop_recall"][0] += stop_pred
                m["S_stop_recall"][1] += 1
                tier[("S", q["difficulty"])][0] += stop_pred
                tier[("S", q["difficulty"])][1] += 1
            else:
                m["S_false_stop"][0] += stop_pred
                m["S_false_stop"][1] += 1
            m["S_label"][0] += r["safety_label"] == q["expected_label"]
            m["S_label"][1] += 1
    return {"m": dict(m), "tier": dict(tier), "calls": calls, "lat": lat, "errors": n_err}


def build_report(questions: list[dict], keys: list[str]) -> str:
    gold = gold_index(questions)
    res = {k: metrics_for(load_rows(k), gold) for k, _, _ in CONFIGS if k in keys}
    labels = {k: lab for k, lab, _ in CONFIGS}

    def cell(r: dict, name: str, as_rate: bool = True) -> str:
        if name not in r["m"]:
            return "—"
        k, n = r["m"][name]
        return fmt_rate(k, n)

    out: list[str] = []
    n_by_group = defaultdict(int)
    for q in questions:
        n_by_group[q["group"]] += 1
    out.append(f"### Bảng ablation — số câu: H={n_by_group['H']}, R={n_by_group['R']}, "
               f"O={n_by_group['O']}, S={n_by_group['S']} (S chỉ chạy các hàng có safety)\n")
    out.append("Mọi tỉ lệ: giá trị [khoảng tin cậy Wilson 95%]. `—` = không áp dụng / chưa chạy.\n")
    head = ["Cấu hình", "H Article@5", "R Article@5", "R chunk-hit", "H must_mention", "H từ chối nhầm ↓",
            "H câu có trích dẫn", "H faithfulness*", "H citation prec.*", "O no-info prec.", "S stop recall", "LLM call/câu", "P95 (s)", "lỗi"]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    for k, _, _ in CONFIGS:
        if k not in res:
            continue
        r = res[k]
        calls = r["calls"]
        out.append("| " + " | ".join([
            labels[k], cell(r, "H_article5"), cell(r, "R_article5"), cell(r, "R_chunk"), cell(r, "H_must"),
            cell(r, "H_false_refusal"), cell(r, "H_cov"), cell(r, "H_faith"), cell(r, "H_cite"), cell(r, "O_noinfo"),
            cell(r, "S_stop_recall"),
            f"{sum(calls) / len(calls):.1f}" if calls else "—", f"{p95(r['lat']):.1f}" if r["lat"] else "—",
            str(r["errors"]),
        ]) + " |")
    out.append("\n\\* LLM-judge, chưa hiệu chuẩn bằng người — chỉ số tham khảo. Faithfulness = 1 − tỉ lệ câu "
               "unsupported, trong đó câu y khoa KHÔNG có [n] cũng tính unsupported, nên phải đọc cùng cột \"câu có "
               "trích dẫn\": faithfulness thấp mà citation precision cao nghĩa là thiếu trích dẫn chứ không phải bịa. "
               "Citation precision: câu `supported` tính 1, `partial` tính 0.5, trên các câu có trích dẫn. Từ chối nhầm: câu trả lời H bị từ chối dù có "
               "bài đúng trong corpus.\n\nP95 đo khi nhiều câu chạy song song (--concurrency, mặc định 4) trên cùng "
               "một LLM endpoint và model server, nên chỉ so sánh tương đối giữa các hàng; muốn số sạch chạy "
               "`--concurrency 1`.\n")

    out.append("\n### Theo độ khó (Article@5; O: no-info precision; S: stop recall)\n")
    cols = [(g, t) for g in ("R", "H") for t in TIERS]
    out.append("| Cấu hình | " + " | ".join(f"{g} {t}" for g, t in cols) + " | " + " | ".join(f"O {t}" for t in TIERS) + " |")
    out.append("|" + "---|" * (1 + len(cols) + len(TIERS)))
    for k, _, _ in CONFIGS:
        if k not in res:
            continue
        t = res[k]["tier"]
        cells = [fmt_rate(*t[c]) if c in t else "—" for c in cols]
        cells += [fmt_rate(*t[("O", x)]) if ("O", x) in t else "—" for x in TIERS]
        out.append(f"| {labels[k]} | " + " | ".join(cells) + " |")

    safety_keys = [k for k, _, f in CONFIGS if f.safety and k in res]
    if safety_keys:
        out.append("\n### Safety (nhóm S, chỉ các hàng bật safety)\n")
        out.append("| Cấu hình | stop recall | dừng nhầm ↓ | độ chính xác nhãn | stop recall: easy | medium | hard |")
        out.append("|---|---|---|---|---|---|---|")
        for k in safety_keys:
            r = res[k]
            tcells = [fmt_rate(*r["tier"][("S", t)]) if ("S", t) in r["tier"] else "—" for t in TIERS]
            out.append(f"| {labels[k]} | {cell(r, 'S_stop_recall')} | {cell(r, 'S_false_stop')} | "
                       f"{cell(r, 'S_label')} | " + " | ".join(tcells) + " |")
    return "\n".join(out)


async def rejudge(key: str, concurrency: int) -> None:
    """Chấm lại faithfulness / citation precision cho các câu H đã cache (không chạy lại pipeline)."""
    path = OUT_DIR / f"{key}.jsonl"
    if not path.exists():
        return
    chunk_by_id = {}
    for line in open("data/processed/chunks.jsonl", encoding="utf-8"):
        c = json.loads(line)
        chunk_by_id[c["chunk_id"]] = c
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    sem = asyncio.Semaphore(concurrency)

    async def one(r: dict) -> None:
        if r["group"] != "H" or "error" in r:
            return
        async with sem:
            r["judge"] = await judge_answer(r["answer"], [chunk_by_id[i] for i in r["chunk_ids"]])

    await asyncio.gather(*(one(r) for r in rows))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{key}] đã chấm lại {sum(r['group'] == 'H' for r in rows)} câu H", flush=True)


def retrieval_questions(groups: list[str], split: str) -> list[dict]:
    """Toàn bộ câu của các nhóm được chọn (không lấy mẫu theo mức) — dùng cho --retrieval-only."""
    files = {"R": ("r_full.jsonl", "R"), "H": ("h_full.jsonl", "H"), "S": ("s_full.jsonl", "S")}
    rows = []
    for g in groups:
        for r in _read(files[g][0]):
            if split == "all" or r.get("split") == split:
                rows.append({**r, "group": files[g][1]})
    return rows


def build_retrieval_report(questions: list[dict], keys: list[str]) -> str:
    gold = gold_index(questions)
    labels = {k: lab for k, lab, _ in CONFIGS}
    labels.update({  # nhãn riêng: ở chế độ này không có generate / validator
        "P1": "P1: hybrid + rerank", "P2": "+ corrective", "P3": "+ routing", "P3A": "+ research agent",
        "P5S": "+ safety", "P5V": "Hệ thống đầy đủ (routing + corrective + agent + safety)",
    })
    out = [f"### Retrieval-only — {len(questions)} câu "
           f"({', '.join(f'{g}={sum(q['group'] == g for q in questions)}' for g in sorted({q['group'] for q in questions}))}); "
           "không sinh câu trả lời\n",
           "Mọi tỉ lệ: giá trị [khoảng tin cậy Wilson 95%]. Context cuối = các chunk sau rerank (tối đa 5).\n"]
    for g in sorted({q["group"] for q in questions}):
        out.append(f"\n#### Nhóm {g}\n")
        if g == "S":  # safety: chỉ có nhãn, không có retrieval để đo
            out.append("| Cấu hình | stop recall | dừng nhầm ↓ | độ chính xác nhãn | LLM call/câu |")
            out.append("|---|---|---|---|---|")
            for k in keys:
                rows = {i: r for i, r in load_rows(k, ".ret").items() if gold.get(i, {}).get("group") == "S"}
                if not rows or not CONFIGS_BY_KEY[k].safety:
                    continue
                res = metrics_for(rows, gold)
                calls = res["calls"]
                cell = lambda n: fmt_rate(*res["m"][n]) if n in res["m"] else "—"  # noqa: E731
                out.append(f"| {labels[k]} | {cell('S_stop_recall')} | {cell('S_false_stop')} | {cell('S_label')} | "
                           f"{sum(calls) / len(calls):.1f} |")
            continue
        head = ["Cấu hình", "Article@5"] + (["chunk-hit"] if g == "R" else []) + [f"A@5 {t}" for t in TIERS] + [
            "dừng bởi safety", "Article@5 (không tính câu bị dừng)", "LLM call/câu", "P95 (s)"]
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "---|" * len(head))
        for k in keys:
            rows = {i: r for i, r in load_rows(k, ".ret").items() if gold.get(i, {}).get("group") == g}
            ok = [r for r in rows.values() if "error" not in r]
            if not ok:
                continue
            art = lambda rs: sum(any(u in gold[r["id"]]["gold_urls"] for u in r["urls"]) for r in rs)  # noqa: E731
            cells = [labels[k], fmt_rate(art(ok), len(ok))]
            if g == "R":
                cells.append(fmt_rate(sum(gold[r["id"]]["gold_chunk_id"] in r["chunk_ids"] for r in ok), len(ok)))
            for t in TIERS:
                rt = [r for r in ok if gold[r["id"]]["difficulty"] == t]
                cells.append(fmt_rate(art(rt), len(rt)) if rt else "—")
            stopped = sum(r["safety_label"] in STOP_LABELS for r in ok)
            calls = [r["llm_calls"] for r in ok]
            go = [r for r in ok if r["safety_label"] not in STOP_LABELS]
            cells += [f"{stopped}/{len(ok)}", fmt_rate(art(go), len(go)), f"{sum(calls) / len(calls):.1f}",
                      f"{p95([r['latency_ms'] / 1000 for r in ok]):.1f}"]
            out.append("| " + " | ".join(cells) + " |")
        errs = sum("error" in r for r in rows.values()) if rows else 0
        if errs:
            out.append(f"\n({errs} câu lỗi, đã loại khỏi bảng)")
    return "\n".join(out)


# ----------------------------------------------------------------------------- main
async def amain(args) -> None:
    if args.retrieval_only:
        wanted = [c for c in CONFIGS if not args.configs or c[0] in args.configs.split(",")]
        questions = retrieval_questions(args.groups.split(","), args.split)
        if args.limit:
            questions = questions[: args.limit]
        if not args.report_only:
            for key, _, features in wanted:
                await run_config(key, dataclasses.replace(features, generate=False), questions,
                                 args.concurrency, with_judge=False, suffix=".ret")
        report = build_retrieval_report(questions, [k for k, _, _ in wanted])
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "report_retrieval.md").write_text(report, encoding="utf-8")
        print("\n" + report)
        print(f"\n(đã lưu {OUT_DIR / 'report_retrieval.md'})")
        return
    if args.rejudge:
        for key, _, _ in CONFIGS:
            if not args.configs or key in args.configs.split(","):
                await rejudge(key, args.concurrency)
        args.report_only = True
    questions = select_questions(args.split, args.r_per_tier, args.limit)
    wanted = [c for c in CONFIGS if not args.configs or c[0] in args.configs.split(",")]
    if not args.report_only:
        for key, _, features in wanted:
            await run_config(key, features, questions, args.concurrency, with_judge=not args.no_judge)
    report = build_report(questions, [k for k, _, _ in wanted])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n(đã lưu {OUT_DIR / 'report.md'})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--configs", help="vd P1,P2 (mặc định: tất cả)")
    ap.add_argument("--r-per-tier", type=int, default=20)
    ap.add_argument("--limit", type=int, help="chỉ lấy N câu đầu mỗi nhóm (chạy thử)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--no-judge", action="store_true", help="bỏ LLM-judge (faithfulness / citation precision)")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="chỉ đo retrieval (graph dừng trước generate), chạy toàn bộ câu của --groups, bỏ qua --r-per-tier")
    ap.add_argument("--groups", default="R", help="với --retrieval-only: R, H, S hoặc kết hợp, vd R,S")
    ap.add_argument("--rejudge", action="store_true", help="chấm lại faithfulness từ câu trả lời đã cache")
    asyncio.run(amain(ap.parse_args()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
