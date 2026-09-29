"""Metric retrieval dùng chung cho eval Phase 1 (mục 6.10)."""
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
