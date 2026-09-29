#!/usr/bin/env python3
"""Canonical: gộp các index_url trỏ cùng một bài (redirect/trùng tên) theo canonical_url.

Usage:
    python -m crawler.canonical --index data/raw/index.jsonl \
        --manifest data/raw/manifest.jsonl --out data/raw/canonical.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

from crawler.http_utils import slug_from_url


def build_canonical_map(index_path: str, manifest_path: str) -> Dict[str, dict]:
    names_by_index_url: Dict[str, str] = {}
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            names_by_index_url[row["index_url"]] = row["index_name"]

    groups: Dict[str, dict] = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["status"] not in ("ok", "cached"):
                continue
            canonical_url = row["canonical_url"]
            name = names_by_index_url.get(row["index_url"], "")

            g = groups.get(canonical_url)
            if g is None:
                g = groups[canonical_url] = {
                    "canonical_url": canonical_url,
                    "article_id": slug_from_url(canonical_url),
                    "aliases": [],
                    "index_urls": [],
                    "html_path": row["html_path"],
                }
            if row["index_url"] not in g["index_urls"]:
                g["index_urls"].append(row["index_url"])
            if name and name not in g["aliases"]:
                g["aliases"].append(name)
    return groups


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/raw/index.jsonl")
    ap.add_argument("--manifest", default="data/raw/manifest.jsonl")
    ap.add_argument("--out", default="data/raw/canonical.jsonl")
    args = ap.parse_args(argv)

    groups = build_canonical_map(args.index, args.manifest)
    with open(args.out, "w", encoding="utf-8") as f:
        for g in groups.values():
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print(f"[done] {len(groups)} canonical articles -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
