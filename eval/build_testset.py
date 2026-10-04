#!/usr/bin/env python3
"""Dựng bộ test phân tầng độ khó cho nhóm R, S, O (thiết kế mục 5) + thống kê toàn bộ bộ test.

  python -m eval.build_testset r [--per-cell 9] [--oversample 1.5] [--limit-cells N]
  python -m eval.build_testset s
  python -m eval.build_testset o
  python -m eval.build_testset stats

Nhóm R: lấy mẫu chunk phân tầng theo section_type x độ khó (mỗi chunk dùng cho đúng 1 mức),
LLM sinh 1 câu hỏi theo luật của mức đó, rồi lọc: (1) không chép nguyên tiêu đề bài,
(2) LLM kiểm tra "trả lời được từ chunk" + "hiểu được khi đứng một mình", (3) khử trùng lặp.
Mức độ khó được kiểm chứng sau bằng số liệu: lexical_overlap (câu hỏi ↔ chunk) phải giảm dần
easy > medium > hard, và Article@5 của baseline retrieval cũng vậy (xem README).

Chia dev/test 40/60 cố định, phân tầng theo (nhóm con, độ khó), không phụ thuộc thời điểm chạy.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

from pydantic import BaseModel
from rapidfuzz import fuzz

from eval.cases_oos import EASY, HARD, MEDIUM_CANDIDATES, MEDIUM_TARGET

try:  # cases_safety.py không commit (xem .gitignore) — dựng lại từ file .example.py
    from eval.cases_safety import ACTION_BY_LABEL, CASES
except ModuleNotFoundError:
    ACTION_BY_LABEL, CASES = {}, []
from medical_agentic_rag.llm.client import run_task
from medical_agentic_rag.llm.tasks import TASKS, TaskSpec
from medical_agentic_rag.safety.rules import STOP_LABELS, rule_label, strip_accents

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
TESTSET_DIR = Path("eval/testset")
CANDIDATE_CACHE = Path("data/eval_r_candidates.jsonl")

SECTION_TYPES = [
    "overview", "symptom", "cause_risk", "diagnosis", "treatment",
    "prevention", "complication_prognosis", "when_to_see_doctor", "care",
]
TIERS = ["easy", "medium", "hard"]
MIN_TOKENS, MAX_TOKENS = 80, 350
MAX_PER_ARTICLE = 2
CONCURRENCY = 10
HARD_NO_ACCENT_RATE = 0.5  # nửa số câu hard bị bỏ dấu bằng code (~17% toàn nhóm R, theo mục 5)

TASKS["eval_gen_question"] = TaskSpec("fast", 0.8, 512)
TASKS["eval_check_question"] = TaskSpec("fast", 0.0, 512)

TIER_RULES = {
    "easy": "Hỏi trực tiếp, nêu tên bệnh / thuật ngữ y khoa đúng như trong đoạn văn, viết đủ dấu, văn viết chuẩn.",
    "medium": (
        "Diễn đạt lại bằng từ khác: dùng tên gọi thông thường hoặc tên gọi khác của bệnh / thuật ngữ, "
        "không dùng lại cụm từ đặc trưng của đoạn văn; vẫn phải nói rõ đang hỏi về bệnh nào."
    ),
    "hard": (
        "Viết như người bệnh / người nhà nói chuyện: kể tình huống cá nhân hoặc mô tả triệu chứng, "
        "văn nói, không dùng thuật ngữ y khoa của đoạn văn. Hạn chế nêu thẳng tên bệnh; người trả lời "
        "phải suy ra chủ đề từ mô tả. Câu vẫn phải đủ cụ thể để tìm đúng chủ đề."
    ),
}

GEN_TEMPLATE = """Bạn viết câu hỏi kiểm thử cho hệ thống hỏi đáp y khoa tiếng Việt.
Dựa vào ĐOẠN VĂN, viết MỘT câu hỏi mà đoạn văn trả lời được.

Bài: {title}
Mục: {section}

ĐOẠN VĂN:
{text}

MỨC ĐỘ "{tier}": {rule}

Quy tắc chung: câu hỏi ngắn gọn, tự nhiên như người dùng thật sẽ gõ (tối đa ~25 từ, không rườm rà);
không chép nguyên tiêu đề bài; không nhắc "đoạn văn", "bài viết"; chỉ hỏi một ý chính;
câu hỏi hiểu được khi đứng một mình (không dùng "bệnh này", "nó" khi chưa nói rõ). Trả về JSON."""

CHECK_TEMPLATE = """ĐOẠN VĂN:
{text}

CÂU HỎI: {question}

Đánh giá:
- answerable: đoạn văn chứa thông tin trả lời trực tiếp ý chính của câu hỏi.
- self_contained: câu hỏi hiểu được và tìm kiếm được mà không cần xem đoạn văn (không phụ thuộc
  đại từ hay ngữ cảnh bị thiếu). Trả về JSON."""


class GeneratedQuestion(BaseModel):
    question: str


class QuestionCheck(BaseModel):
    answerable: bool
    self_contained: bool


# ----------------------------------------------------------------------------- tiện ích
def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", strip_accents(text)) if len(t) >= 2]


STOP = frozenset("la gi co khong nhu the nao bi cua va cho nen minh toi ban nhung thi hay duoc mot cac".split())


def lexical_overlap(question: str, chunk_text: str, title: str) -> float:
    """Tỉ lệ từ nội dung của câu hỏi xuất hiện trong chunk + tiêu đề (đã bỏ dấu)."""
    q = [t for t in _tokens(question) if t not in STOP]
    if not q:
        return 0.0
    pool = set(_tokens(chunk_text)) | set(_tokens(title))
    return round(sum(t in pool for t in q) / len(q), 3)


def leaks_title(question: str, title: str) -> bool:
    q, t = strip_accents(question), strip_accents(title)
    q = re.sub(r"[^\w\s]", " ", q)
    t = re.sub(r"[^\w\s]", " ", t)
    q, t = re.sub(r"\s+", " ", q).strip(), re.sub(r"\s+", " ", t).strip()
    return t in q or fuzz.ratio(q, t) > 85


def assign_split(rows: list[dict], strata_key) -> None:
    """dev/test 40/60 cố định, phân tầng: duyệt các tầng theo thứ tự, trong tầng xếp theo id, dùng
    một bộ đếm toàn cục (Bresenham) nên tỉ lệ dev luôn sát 40% kể cả khi tầng chỉ có 1-3 câu."""
    strata: dict = defaultdict(list)
    for r in rows:
        strata[strata_key(r)].append(r)
    k = 0
    for key in sorted(strata, key=str):
        for r in sorted(strata[key], key=lambda r: r["id"]):
            r["split"] = "dev" if int((k + 1) * 0.4) > int(k * 0.4) else "test"
            k += 1


def write_jsonl(name: str, rows: list[dict]) -> Path:
    path = TESTSET_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_chunks() -> list[dict]:
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------------------------------- nhóm R
def sample_cells(chunks: list[dict], n_per_cell: int, seed: int = 42) -> dict[tuple[str, str], list[dict]]:
    """Mỗi ô (section_type, tier) lấy n_per_cell chunk ứng viên; chunk không dùng lại giữa các ô,
    mỗi bài tối đa MAX_PER_ARTICLE chunk trong toàn bộ nhóm R."""
    rng = random.Random(seed)
    eligible: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        if c["chunk_type"] == "text" and MIN_TOKENS <= c["n_tokens"] <= MAX_TOKENS and c["section_type"] in SECTION_TYPES:
            eligible[c["section_type"]].append(c)
    per_article: Counter = Counter()
    cells: dict[tuple[str, str], list[dict]] = {}
    for st in SECTION_TYPES:
        pool = sorted(eligible[st], key=lambda c: c["chunk_id"])
        rng.shuffle(pool)
        it = iter(pool)
        quota = min(n_per_cell, len(pool) // len(TIERS))  # pool nhỏ (vd. care): chia đều cho 3 mức
        for tier in TIERS:
            picked: list[dict] = []
            for c in it:
                if per_article[c["article_id"]] >= MAX_PER_ARTICLE:
                    continue
                per_article[c["article_id"]] += 1
                picked.append(c)
                if len(picked) >= quota:
                    break
            cells[(st, tier)] = picked
    return cells


def _load_cache() -> dict[str, dict]:
    if not CANDIDATE_CACHE.exists():
        return {}
    with open(CANDIDATE_CACHE, encoding="utf-8") as f:
        return {r["key"]: r for r in (json.loads(l) for l in f if l.strip())}


async def _make_candidate(chunk: dict, tier: str, sem: asyncio.Semaphore, cache: dict) -> dict | None:
    key = f"{chunk['chunk_id']}|{tier}"
    if key in cache:
        return cache[key]
    section = " > ".join(chunk.get("section_path") or []) or chunk["section_type"]
    async with sem:
        try:
            gen = await run_task(
                "eval_gen_question",
                GEN_TEMPLATE.format(title=chunk["title"], section=section, text=chunk["text"], tier=tier, rule=TIER_RULES[tier]),
                {"llm_calls": 0, "trace_id": "eval-testset"}, schema=GeneratedQuestion,
            )
            question = gen.question.strip()
            check = await run_task(
                "eval_check_question",
                CHECK_TEMPLATE.format(text=chunk["text"], question=question),
                {"llm_calls": 0, "trace_id": "eval-testset"}, schema=QuestionCheck,
            )
        except Exception as exc:  # LLMTaskFailed, mạng... : bỏ ứng viên, không làm hỏng cả lượt chạy
            print(f"  [skip] {key}: {type(exc).__name__}", file=sys.stderr)
            return None
    row = {
        "key": key, "chunk_id": chunk["chunk_id"], "tier": tier, "question": question,
        "answerable": check.answerable, "self_contained": check.self_contained,
    }
    with open(CANDIDATE_CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    cache[key] = row
    return row


async def build_r(per_cell: int, oversample: float, limit_cells: int | None) -> list[dict]:
    chunks = load_chunks()
    cells = sample_cells(chunks, max(1, round(per_cell * oversample)))
    items = list(cells.items())[:limit_cells] if limit_cells else list(cells.items())

    CANDIDATE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache, sem = _load_cache(), asyncio.Semaphore(CONCURRENCY)
    jobs = [(cell, c) for cell, cs in items for c in cs]
    print(f"[R] {len(items)} ô, {len(jobs)} ứng viên (đã cache: {sum(f'{c['chunk_id']}|{cell[1]}' in cache for cell, c in jobs)})")
    results = await asyncio.gather(*(_make_candidate(c, cell[1], sem, cache) for cell, c in jobs))

    rows: list[dict] = []
    seen_questions: list[str] = []
    rng = random.Random(7)
    reject = Counter()
    for (cell, chunk), cand in zip(jobs, results):
        if cand is None:
            reject["llm_error"] += 1
            continue
        st, tier = cell
        if sum(1 for r in rows if r["section_type"] == st and r["difficulty"] == tier) >= per_cell:
            continue
        q = cand["question"]
        if not (cand["answerable"] and cand["self_contained"]):
            reject["check"] += 1
            continue
        if leaks_title(q, chunk["title"]):
            reject["title_leak"] += 1
            continue
        if any(fuzz.ratio(strip_accents(q), strip_accents(o)) > 90 for o in seen_questions):
            reject["duplicate"] += 1
            continue
        no_accent = tier == "hard" and rng.random() < HARD_NO_ACCENT_RATE
        if no_accent:
            q = strip_accents(q).replace("?", "").strip()
        seen_questions.append(q)
        rows.append({
            "question": q, "difficulty": tier, "section_type": st,
            "gold_chunk_id": chunk["chunk_id"], "gold_urls": [chunk["url"]], "article_id": chunk["article_id"],
            "no_diacritics": no_accent,
            "lexical_overlap": lexical_overlap(q, chunk["text"], chunk["title"]),
        })

    rows.sort(key=lambda r: (r["section_type"], TIERS.index(r["difficulty"]), r["gold_chunk_id"]))
    for i, r in enumerate(rows, 1):
        r["id"] = f"R-{i:03d}"
        r["group"] = "R"
    assign_split(rows, lambda r: (r["section_type"], r["difficulty"]))
    path = write_jsonl("r_full.jsonl", rows)
    print(f"[R] {len(rows)} câu -> {path}; loại: {dict(reject)}")
    short = [(k, per_cell - n) for k, n in Counter((r["section_type"], r["difficulty"]) for r in rows).items() if n < per_cell]
    if short or len(rows) < len(items) * per_cell:
        print(f"[R] cảnh báo: có ô chưa đủ {per_cell} câu — tăng --oversample (xem `stats`).")
    return rows


# ----------------------------------------------------------------------------- nhóm S, O
def build_s() -> list[dict]:
    if not CASES:
        print("[S] thiếu eval/cases_safety.py (không commit trong repo). Dựng lại bằng:\n"
              "    cp eval/cases_safety.example.py eval/cases_safety.py\n"
              "rồi viết thêm ca cho đủ 6 nhãn — xem hướng dẫn trong chính file đó.", file=sys.stderr)
        return []
    rows = []
    for i, (question, label, difficulty, note) in enumerate(CASES, 1):
        rows.append({
            "id": f"S-{i:03d}", "group": "S", "question": question, "expected_label": label,
            "expected_action": ACTION_BY_LABEL[label], "difficulty": difficulty, "note": note,
            "no_diacritics": strip_accents(question) == question.lower().replace("đ", "d"),
        })
    assign_split(rows, lambda r: (r["expected_label"], r["difficulty"]))
    path = write_jsonl("s_full.jsonl", rows)
    stop = [r for r in rows if r["expected_action"] == "stop"]
    caught = sum(rule_label(r["question"]) in STOP_LABELS for r in stop)
    print(f"[S] {len(rows)} câu -> {path}; nhóm 'stop': lớp 1 (rule) tự bắt {caught}/{len(stop)}, "
          f"còn lại phụ thuộc lớp 2 (LLM)")
    return rows


def _corpus_text_norm() -> str:
    parts = []
    for c in load_chunks():
        parts.append(c["title"])
        parts.extend(c.get("aliases") or [])
        parts.append(c["text"])
    return strip_accents("\n".join(parts))


def build_o() -> list[dict]:
    corpus = _corpus_text_norm()
    medium = []
    for terms, question in MEDIUM_CANDIDATES:
        present = [t for t in terms if re.search(rf"(?<!\w){re.escape(strip_accents(t))}", corpus)]
        if present:
            print(f"  [O] loại {terms}: có trong corpus ({', '.join(present)})")
        else:
            medium.append(question)
    medium = medium[:MEDIUM_TARGET]
    rows = []
    spec = (
        [(q, "easy", "decline_out_of_scope") for q in EASY]
        + [(q, "medium", "no_info") for q in medium]
        + [(q, "hard", "either") for q in HARD]
    )
    for i, (question, difficulty, behavior) in enumerate(spec, 1):
        rows.append({"id": f"O-{i:03d}", "group": "O", "question": question,
                     "difficulty": difficulty, "expected_behavior": behavior})
    assign_split(rows, lambda r: r["difficulty"])
    path = write_jsonl("o_full.jsonl", rows)
    print(f"[O] {len(rows)} câu -> {path} (medium giữ {len(medium)}/{len(MEDIUM_CANDIDATES)} ứng viên)")
    if len(medium) < MEDIUM_TARGET:
        print(f"[O] cảnh báo: chỉ {len(medium)}/{MEDIUM_TARGET} câu medium — thêm ứng viên vào eval/cases_oos.py")
    return rows


# ----------------------------------------------------------------------------- thống kê
def stats() -> None:
    files = sorted(TESTSET_DIR.glob("*_full.jsonl"))
    print(f"{'file':20s} {'n':>4s}  " + "  ".join(f"{t:>6s}" for t in TIERS) + "   dev/test   no-dấu")
    for f in files:
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        diff = Counter(r.get("difficulty") for r in rows)
        split = Counter(r.get("split") or "-" for r in rows)
        nodia = sum(bool(r.get("no_diacritics")) for r in rows)
        print(f"{f.name:20s} {len(rows):4d}  " + "  ".join(f"{diff.get(t, 0):6d}" for t in TIERS)
              + f"   {split.get('dev', 0)}/{split.get('test', 0):<3d}    {nodia}")
    r_path = TESTSET_DIR / "r_full.jsonl"
    if r_path.exists():
        rows = [json.loads(l) for l in open(r_path, encoding="utf-8") if l.strip()]
        print("\nR: lexical_overlap trung bình theo mức (kỳ vọng giảm dần):")
        for t in TIERS:
            vals = [r["lexical_overlap"] for r in rows if r["difficulty"] == t]
            if vals:
                print(f"  {t:7s} {sum(vals) / len(vals):.3f}  (n={len(vals)})")
        cell = Counter((r["section_type"], r["difficulty"]) for r in rows)
        print("R: số câu theo section_type x mức:")
        for st in SECTION_TYPES:
            print(f"  {st:24s} " + "  ".join(f"{cell.get((st, t), 0):3d}" for t in TIERS))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("r")
    r.add_argument("--per-cell", type=int, default=9)
    r.add_argument("--oversample", type=float, default=1.5)
    r.add_argument("--limit-cells", type=int, default=None, help="chỉ chạy N ô đầu (chạy thử)")
    sub.add_parser("s")
    sub.add_parser("o")
    sub.add_parser("stats")
    args = ap.parse_args()
    if args.cmd == "r":
        asyncio.run(build_r(args.per_cell, args.oversample, args.limit_cells))
    elif args.cmd == "s":
        build_s()
    elif args.cmd == "o":
        build_o()
    else:
        stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
