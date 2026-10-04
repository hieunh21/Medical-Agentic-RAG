"""Metric và helper thống kê dùng chung cho mọi script trong eval/."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence


def article_at_k(ranked_urls: Sequence[str], gold_urls: Sequence[str], k: int = 5) -> int:
    top = ranked_urls[:k]
    return int(any(u in gold_urls for u in top))


def _rank_of(gold_id: str, ranked_ids: Sequence[str]) -> Optional[int]:
    for i, rid in enumerate(ranked_ids, 1):
        if rid == gold_id:
            return i
    return None


def recall_at_k(gold_chunk_id: str, ranked_chunk_ids: Sequence[str], k: int = 10) -> int:
    return int(gold_chunk_id in ranked_chunk_ids[:k])


def mrr_at_k(gold_chunk_id: str, ranked_chunk_ids: Sequence[str], k: int = 10) -> float:
    rank = _rank_of(gold_chunk_id, ranked_chunk_ids[:k])
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(gold_chunk_id: str, ranked_chunk_ids: Sequence[str], k: int = 10) -> float:
    rank = _rank_of(gold_chunk_id, ranked_chunk_ids[:k])
    return 1.0 / math.log2(rank + 1) if rank else 0.0


# --------------------------------------------------------------------------- #
# Helper thống kê: n trong các bộ test ở đây nhỏ, mọi tỉ lệ báo cáo kèm khoảng tin cậy.
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy Wilson cho tỉ lệ k/n — ổn định hơn Wald khi n nhỏ hoặc p gần 0/1."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_rate(k: float, n: int) -> str:
    """"0.93 [0.89-0.96]" — k có thể là số thực (vd. điểm 0.5 cho câu partial)."""
    if n == 0:
        return "—"
    lo, hi = wilson(round(k), n)
    return f"{k / n:.2f} [{lo:.2f}–{hi:.2f}]"


def p95(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1)]
