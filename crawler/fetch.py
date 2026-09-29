#!/usr/bin/env python3
"""Fetch: tải HTML cho từng index_url, lưu cache + manifest.

Usage:
    python -m crawler.fetch --index data/raw/index.jsonl --workers 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import List, Optional

from bs4 import BeautifulSoup

from crawler.http_utils import fetch, make_session, polite_sleep, slug_from_url


def extract_canonical(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    link = soup.select_one("link[rel=canonical]")
    href = (link.get("href") or "").strip() if link else ""
    return href or None


def fetch_one(index_url: str, out_html_dir: str, overwrite: bool) -> dict:
    slug = slug_from_url(index_url)
    html_path = os.path.join(out_html_dir, f"{slug}.html")

    if os.path.exists(html_path) and not overwrite:
        html = open(html_path, encoding="utf-8").read()
        return {
            "index_url": index_url,
            "final_url": index_url,
            "canonical_url": extract_canonical(html) or index_url,
            "status": "cached",
            "fetched_at": None,
            "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
            "html_path": html_path,
        }

    polite_sleep()
    session = make_session()
    resp = fetch(session, index_url)
    html = resp.text
    os.makedirs(out_html_dir, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return {
        "index_url": index_url,
        "final_url": resp.url,
        "canonical_url": extract_canonical(html) or resp.url,
        "status": "ok",
        "fetched_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
        "html_path": html_path,
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/raw/index.jsonl")
    ap.add_argument("--html-out", default="data/raw/html")
    ap.add_argument("--manifest-out", default="data/raw/manifest.jsonl")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    with open(args.index, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        items = items[: args.limit]

    os.makedirs(args.html_out, exist_ok=True)
    results: List[dict] = []
    ok = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(fetch_one, it["index_url"], args.html_out, args.overwrite): it["index_url"]
            for it in items
        }
        for i, fut in enumerate(as_completed(futures), 1):
            url = futures[fut]
            try:
                row = fut.result()
                results.append(row)
                ok += 1
                print(f"[OK] {i}/{len(items)} {row['status']} {slug_from_url(url)}", file=sys.stderr)
            except Exception as exc:
                err += 1
                print(f"[ERR] {i}/{len(items)} {url}: {exc}", file=sys.stderr)

    with open(args.manifest_out, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[done] ok={ok} err={err} -> {args.manifest_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
