#!/usr/bin/env python3
"""Ingest: chunks.jsonl -> embed (dense qua model server) + sparse (vi_sparse) -> Qdrant.

Usage:
    python -m ingestion.ingest --chunks data/processed/chunks.jsonl --recreate
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

from ingestion.embed_client import embed_documents
from ingestion.qdrant_store import QDRANT_COLLECTION, create_collection, get_client, point_id, upsert_chunks
from ingestion.vi_sparse import doc_sparse, doc_term_count

BATCH_SIZE = 16


def load_chunks(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def existing_point_ids(client) -> set[str]:
    """Toàn bộ point id đã có trong collection — dùng để resume, bỏ qua chunk đã ingest."""
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            QDRANT_COLLECTION, limit=2000, with_payload=False, with_vectors=False, offset=offset,
        )
        ids.update(str(p.id) for p in points)
        if offset is None:
            break
    return ids


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--recreate", action="store_true", help="Xoá sạch collection, ingest lại từ đầu")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args(argv)

    chunks = load_chunks(args.chunks)
    print(f"[i] {len(chunks)} chunks", file=sys.stderr)

    # avg_len cho BM25 saturation — cần 1 lượt qua toàn corpus trước.
    lengths = [doc_term_count(c["text"]) for c in chunks]
    avg_len = sum(lengths) / len(lengths) if lengths else 1.0
    print(f"[i] avg_len={avg_len:.1f}", file=sys.stderr)

    client = get_client()
    collection_existed = client.collection_exists(QDRANT_COLLECTION)
    create_collection(client, recreate=args.recreate)

    if not args.recreate and collection_existed:
        done_ids = existing_point_ids(client)
        before = len(chunks)
        chunks = [c for c in chunks if point_id(c["chunk_id"]) not in done_ids]
        print(f"[i] resume: bỏ qua {before - len(chunks)} chunk đã ingest, còn {len(chunks)}", file=sys.stderr)

    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i : i + args.batch_size]
        texts = [c["text"] for c in batch]
        titles = [c["title"] for c in batch]
        dense_vecs = embed_documents(texts, titles)
        sparse_vecs = [doc_sparse(t, avg_len) for t in texts]
        upsert_chunks(client, batch, dense_vecs, sparse_vecs)
        print(f"[OK] {min(i + args.batch_size, len(chunks))}/{len(chunks)}", file=sys.stderr)

    print("[done] ingest complete", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
