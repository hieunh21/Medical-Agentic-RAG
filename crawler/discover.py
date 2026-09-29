#!/usr/bin/env python3
"""Discover: đọc trang A-Z index YouMed -> data/raw/index.jsonl.

Usage:
    python -m crawler.discover --out data/raw/index.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, TypedDict

from bs4 import BeautifulSoup

from crawler.http_utils import fetch, make_session

INDEX_URL = "https://youmed.vn/tin-tuc/trieu-chung-benh/"


class IndexItem(TypedDict):
    index_name: str
    index_letter: str
    index_url: str


def parse_index(html: str) -> List[IndexItem]:
    soup = BeautifulSoup(html, "lxml")
    items: List[IndexItem] = []
    for section in soup.select("#a-z-listing-1 .letter-section"):
        letter_el = section.select_one("h2.letter-title")
        letter = letter_el.get_text(strip=True) if letter_el else ""
        for a in section.select("ul li a[href]"):
            href = a.get("href", "").strip()
            if not href.startswith("http") or "youmed.vn/tin-tuc/" not in href:
                continue
            name = a.get_text(strip=True)
            items.append({"index_name": name, "index_letter": letter, "index_url": href})
    return items


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=INDEX_URL)
    ap.add_argument("--out", default="data/raw/index.jsonl")
    args = ap.parse_args(argv)

    session = make_session()
    print(f"[i] Fetching index: {args.index}", file=sys.stderr)
    html = fetch(session, args.index).text
    items = parse_index(html)
    print(f"[i] Found {len(items)} items", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[done] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
