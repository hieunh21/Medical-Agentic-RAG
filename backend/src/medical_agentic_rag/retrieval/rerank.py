"""Rerank bằng cross-encoder, giới hạn tối đa N chunk/bài, lấy top K vào context."""
from __future__ import annotations

from typing import List

from ingestion.embed_client import rerank as rerank_scores

from medical_agentic_rag.config import settings


def rerank_and_select(question: str, candidates: List[dict]) -> List[dict]:
    if not candidates:
        return []
    scores = rerank_scores(question, [c["text"] for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top = ranked[: settings.RERANK_TOP_K]

    per_article: dict[str, int] = {}
    selected: List[dict] = []
    for chunk, score in top:
        aid = chunk["article_id"]
        if per_article.get(aid, 0) >= settings.MAX_CHUNKS_PER_ARTICLE:
            continue
        per_article[aid] = per_article.get(aid, 0) + 1
        selected.append({**chunk, "rerank_score": float(score)})
        if len(selected) >= settings.FINAL_CONTEXT_K:
            break
    return selected
