#!/usr/bin/env python3
"""Chép collection từ Qdrant local (docker) lên Qdrant Cloud — giữ nguyên vector, không embed lại.

Đọc từ .env:
    QDRANT_URL            nguồn (mặc định http://127.0.0.1:6333)
    QDRANT_CLOUD_URL      đích, dạng https://<id>.<vùng>.cloud.qdrant.io:6333
    QDRANT_CLOUD_API_KEY  API key của cluster đích

Usage (từ repo root, Qdrant local đang chạy):
    python -m deploy.migrate_qdrant            # tạo collection nếu chưa có rồi upsert
    python -m deploy.migrate_qdrant --recreate # xoá collection đích, chép lại từ đầu

Sau khi chép, so top-5 của cùng một truy vấn hybrid trên hai bên để chắc chắn index giống nhau
(sparse dùng Modifier.IDF: IDF do Qdrant đích tự tính lại từ đúng tập điểm đã chép).
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient, models  # noqa: E402

from ingestion.qdrant_store import QDRANT_COLLECTION, create_collection, hybrid_query  # noqa: E402
from ingestion.vi_sparse import query_sparse  # noqa: E402

BATCH = 256
CHECK_QUERIES = ["bệnh addison", "triệu chứng sốt xuất huyết", "cach phong benh zona"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args(argv)

    cloud_url, cloud_key = os.getenv("QDRANT_CLOUD_URL"), os.getenv("QDRANT_CLOUD_API_KEY")
    if not cloud_url or not cloud_key:
        print("Thiếu QDRANT_CLOUD_URL hoặc QDRANT_CLOUD_API_KEY trong .env", file=sys.stderr)
        return 1

    src = QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"), timeout=60)
    dst = QdrantClient(url=cloud_url, api_key=cloud_key, timeout=60)
    total = src.count(QDRANT_COLLECTION, exact=True).count
    print(f"[i] nguồn: {total} điểm trong '{QDRANT_COLLECTION}'", file=sys.stderr)

    create_collection(dst, recreate=args.recreate)

    offset, done = None, 0
    while True:
        points, offset = src.scroll(
            QDRANT_COLLECTION, limit=BATCH, offset=offset, with_payload=True, with_vectors=True,
        )
        if points:
            dst.upsert(QDRANT_COLLECTION, points=[
                models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points
            ])
            done += len(points)
            print(f"[OK] {done}/{total}", file=sys.stderr)
        if offset is None:
            break

    copied = dst.count(QDRANT_COLLECTION, exact=True).count
    print(f"[i] đích: {copied} điểm", file=sys.stderr)
    if copied != total:
        print("[!] số điểm hai bên KHÁC nhau", file=sys.stderr)
        return 1

    # Kiểm chứng: cùng vector truy vấn -> cùng TẬP 30 ứng viên ở hai bên (đúng lượng hybrid_retrieve đưa
    # vào reranker). Không so THỨ TỰ: hai phiên bản Qdrant khác nhau (vd 1.11 local, 1.19 cloud) xếp các
    # điểm RRF bằng nhau khác nhau, mà thứ tự đó đằng nào cũng bị reranker sắp lại.
    probe = src.scroll(QDRANT_COLLECTION, limit=1, with_vectors=["dense"])[0][0].vector["dense"]
    mismatched = 0
    for q in CHECK_QUERIES:
        a = {p.id for p in hybrid_query(src, probe, query_sparse(q), limit=30)}
        b = {p.id for p in hybrid_query(dst, probe, query_sparse(q), limit=30)}
        ok = a == b
        mismatched += not ok
        print(f"[{'OK' if ok else '!!'}] '{q}': {len(a & b)}/30 ứng viên chung", file=sys.stderr)
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
