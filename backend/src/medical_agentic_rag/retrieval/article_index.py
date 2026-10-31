"""Index title+aliases của toàn bộ bài giữ trong bộ nhớ — khớp tên bài bằng code (mục 8.3).

Không cần LLM, chạy vài mili giây. Build 1 lần, từ một trong hai nguồn:
  1. frontmatter của data/processed/articles/ (dev: có đủ related_urls cho tool related_articles);
  2. payload trong Qdrant khi thư mục trên không có — trường hợp deploy: nội dung bài YouMed có bản
     quyền nên không nằm trong repo, nhưng mỗi chunk đã mang article_id/title/aliases/url.
     related_urls không có trong payload nên tool related_articles của agent trả rỗng.
"""
from __future__ import annotations

import glob
import os
import unicodedata
from functools import lru_cache
from typing import Optional

import yaml
from rapidfuzz import fuzz, process

ARTICLES_DIR = os.getenv("ARTICLES_DIR", "data/processed/articles")
MATCH_THRESHOLD = 75


def _normalize(s: str) -> str:
    s = s.lower().replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


class ArticleEntry:
    __slots__ = ("article_id", "title", "aliases", "canonical_url", "related_urls")

    def __init__(self, article_id: str, title: str, aliases: list[str], canonical_url: str, related_urls: list[str]):
        self.article_id = article_id
        self.title = title
        self.aliases = aliases
        self.canonical_url = canonical_url
        self.related_urls = related_urls


def _entries_from_files(paths: list[str]) -> list[ArticleEntry]:
    entries = []
    for path in paths:
        raw = open(path, encoding="utf-8").read()
        fm = yaml.safe_load(raw.split("---", 2)[1]) or {}
        entries.append(ArticleEntry(
            article_id=fm["article_id"], title=fm["title"],
            aliases=fm.get("aliases") or [], canonical_url=fm["canonical_url"],
            related_urls=fm.get("related_urls") or [],
        ))
    return entries


def _entries_from_qdrant() -> list[ArticleEntry]:
    """Một entry mỗi article_id, lấy từ payload chunk. Lỗi Qdrant thì ném ra: lru_cache không lưu
    exception, nên lần gọi sau thử lại thay vì kẹt với một index rỗng suốt vòng đời process."""
    from ingestion.qdrant_store import QDRANT_COLLECTION, get_client  # import muộn: tránh vòng import

    client = get_client()
    by_id: dict[str, ArticleEntry] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            QDRANT_COLLECTION, limit=2000, offset=offset, with_vectors=False,
            with_payload=["article_id", "title", "aliases", "url"],
        )
        for p in points:
            pl = p.payload or {}
            if pl.get("article_id") and pl["article_id"] not in by_id:
                by_id[pl["article_id"]] = ArticleEntry(
                    article_id=pl["article_id"], title=pl.get("title") or "",
                    aliases=pl.get("aliases") or [], canonical_url=pl.get("url") or "", related_urls=[],
                )
        if offset is None:
            break
    return sorted(by_id.values(), key=lambda e: e.article_id)


@lru_cache(maxsize=1)
def _load_index() -> tuple[list[ArticleEntry], dict[str, int]]:
    """Trả về (entries, name_norm -> index trong entries) — mỗi (title hoặc alias) một dòng tra."""
    paths = sorted(glob.glob(f"{ARTICLES_DIR}/*.md"))
    entries = _entries_from_files(paths) if paths else _entries_from_qdrant()
    lookup: dict[str, int] = {}
    for idx, entry in enumerate(entries):
        for name in [entry.title, *entry.aliases]:
            lookup[_normalize(name)] = idx
    return entries, lookup


def warm() -> int:
    """Dựng index ngay lúc khởi động (gọi từ lifespan của API) — để câu hỏi đầu tiên không phải chờ."""
    return len(_load_index()[0])


@lru_cache(maxsize=1)
def _by_canonical_url() -> dict[str, int]:
    entries, _ = _load_index()
    return {e.canonical_url.rstrip("/"): i for i, e in enumerate(entries)}


def _entry_dict(entry: ArticleEntry, score: Optional[float] = None) -> dict:
    d = {"article_id": entry.article_id, "title": entry.title, "aliases": entry.aliases}
    if score is not None:
        d["score"] = score
    return d


def get_titles(article_ids: list[str]) -> list[str]:
    entries, _ = _load_index()
    by_id = {e.article_id: e.title for e in entries}
    return [by_id[a] for a in article_ids if a in by_id]


def get_article(article_id: str) -> Optional[ArticleEntry]:
    entries, _ = _load_index()
    for e in entries:
        if e.article_id == article_id:
            return e
    return None


def find_article(name: str, threshold: int = MATCH_THRESHOLD) -> Optional[dict]:
    """So khớp mờ `name` với title/aliases của ~613 bài. Trả về None nếu không đủ điểm."""
    matches = find_articles(name, k=1, threshold=threshold)
    return matches[0] if matches else None


def find_articles(name: str, k: int = 5, threshold: int = MATCH_THRESHOLD) -> list[dict]:
    """Như find_article nhưng trả về tối đa k kết quả (dùng cho tool find_article của agent)."""
    entries, lookup = _load_index()
    if not lookup:
        return []
    norm = _normalize(name)
    matches = process.extract(norm, lookup.keys(), scorer=fuzz.WRatio, score_cutoff=threshold, limit=k * 3)
    seen_ids: set[str] = set()
    out: list[dict] = []
    for matched_name, score, _ in matches:
        entry = entries[lookup[matched_name]]
        if entry.article_id in seen_ids:
            continue
        seen_ids.add(entry.article_id)
        out.append(_entry_dict(entry, score))
        if len(out) >= k:
            break
    return out


def get_related_articles(article_id: str) -> list[dict]:
    entry = get_article(article_id)
    if entry is None:
        return []
    by_url = _by_canonical_url()
    entries, _ = _load_index()
    out: list[dict] = []
    seen: set[str] = set()
    for url in entry.related_urls:
        idx = by_url.get(url.rstrip("/"))
        if idx is None:
            continue
        rel = entries[idx]
        if rel.article_id in seen or rel.article_id == article_id:
            continue
        seen.add(rel.article_id)
        out.append({"article_id": rel.article_id, "title": rel.title})
    return out
