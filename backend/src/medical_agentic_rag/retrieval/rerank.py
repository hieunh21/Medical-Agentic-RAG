"""Rerank bằng cross-encoder, giới hạn tối đa N chunk/bài, lấy top K vào context."""
from __future__ import annotations

from typing import List

from ingestion.embed_client import rerank as rerank_scores

from medical_agentic_rag.config import settings


def rerank_balanced_by_group(question: str, grouped: dict[str, List[dict]]) -> List[dict]:
    """Như rerank_and_select, nhưng đảm bảo mỗi nhóm (subquery) có ít nhất 1 chunk
    trong kết quả cuối — tránh rerank theo câu hỏi gốc đè hết chunk của 1 nhóm khi
    gộp evidence từ nhiều subquery (mục 8.4, comparison/multi_aspect)."""
    all_candidates = [c for group in grouped.values() for c in group]
    if not all_candidates:
        return []
    scores = rerank_scores(question, [c["text"] for c in all_candidates])
    score_by_id = {c["chunk_id"]: s for c, s in zip(all_candidates, scores)}

    sorted_groups = {
        key: sorted(group, key=lambda c: score_by_id[c["chunk_id"]], reverse=True)
        for key, group in grouped.items() if group
    }
    group_idx = {key: 0 for key in sorted_groups}
    group_keys = list(sorted_groups.keys())

    per_article: dict[str, int] = {}
    chosen_ids: set[str] = set()
    selected: List[dict] = []

    while len(selected) < settings.FINAL_CONTEXT_K and group_keys:
        progressed = False
        for key in list(group_keys):
            items = sorted_groups[key]
            idx = group_idx[key]
            while idx < len(items):
                c = items[idx]
                idx += 1
                if c["chunk_id"] in chosen_ids or per_article.get(c["article_id"], 0) >= settings.MAX_CHUNKS_PER_ARTICLE:
                    continue
                chosen_ids.add(c["chunk_id"])
                per_article[c["article_id"]] = per_article.get(c["article_id"], 0) + 1
                selected.append({**c, "rerank_score": float(score_by_id[c["chunk_id"]])})
                progressed = True
                break
            group_idx[key] = idx
            if idx >= len(items):
                group_keys.remove(key)
            if len(selected) >= settings.FINAL_CONTEXT_K:
                break
        if not progressed:
            break
    return selected


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
