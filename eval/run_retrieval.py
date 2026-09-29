#!/usr/bin/env python3
"""Baseline retrieval eval (mục 6.10): so sánh dense / sparse / hybrid / hybrid+rerank.

Testset JSONL, mỗi dòng:
    {"id": "...", "question": "...", "gold_urls": [...]}          # nhóm H -> Article@5
    {"id": "...", "question": "...", "gold_chunk_id": "..."}      # nhóm R -> Recall/MRR/nDCG@10

Usage:
    python -m eval.run_retrieval --testset eval/testset/example_h.jsonl --mode hybrid_rerank
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import models

from eval.metrics import article_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from ingestion.embed_client import embed_query, rerank as rerank_scores
from ingestion.qdrant_store import QDRANT_COLLECTION, get_client
from ingestion.vi_sparse import query_sparse

MODES = ["dense", "sparse", "hybrid", "hybrid_rerank"]


def search(client, question: str, mode: str, top_k: int = 30) -> List[dict]:
    dense_vec = embed_query(question)
    s_idx, s_val = query_sparse(question)

    if mode == "dense":
        points = client.query_points(
            QDRANT_COLLECTION, query=dense_vec, using="dense", limit=top_k, with_payload=True,
        ).points
    elif mode == "sparse":
        points = client.query_points(
            QDRANT_COLLECTION, query=models.SparseVector(indices=s_idx, values=s_val),
            using="sparse", limit=top_k, with_payload=True,
        ).points
    else:  # hybrid, hybrid_rerank
        points = client.query_points(
            QDRANT_COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=40),
                models.Prefetch(query=models.SparseVector(indices=s_idx, values=s_val), using="sparse", limit=40),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k, with_payload=True,
        ).points

    chunks = [p.payload for p in points]
    if mode == "hybrid_rerank" and chunks:
        scores = rerank_scores(question, [c["text"] for c in chunks])
        chunks = [c for c, _ in sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)]
    return chunks


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True)
    ap.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    args = ap.parse_args(argv)

    with open(args.testset, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    client = get_client()
    article5, recall10, mrr10, ndcg10 = [], [], [], []

    for row in rows:
        chunks = search(client, row["question"], args.mode)
        if "gold_urls" in row:
            ranked_urls = []
            for c in chunks:
                if c["url"] not in ranked_urls:
                    ranked_urls.append(c["url"])
            article5.append(article_at_k(ranked_urls, row["gold_urls"], k=5))
        if "gold_chunk_id" in row:
            ids = [c["chunk_id"] for c in chunks]
            recall10.append(recall_at_k(row["gold_chunk_id"], ids, k=10))
            mrr10.append(mrr_at_k(row["gold_chunk_id"], ids, k=10))
            ndcg10.append(ndcg_at_k(row["gold_chunk_id"], ids, k=10))

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"mode={args.mode} n={len(rows)}")
    if article5:
        print(f"  Article@5  = {avg(article5):.3f}  (n={len(article5)})")
    if recall10:
        print(f"  Recall@10  = {avg(recall10):.3f}")
        print(f"  MRR@10     = {avg(mrr10):.3f}")
        print(f"  nDCG@10    = {avg(ndcg10):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
