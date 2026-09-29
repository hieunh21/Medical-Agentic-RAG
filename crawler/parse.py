#!/usr/bin/env python3
"""Parse: HTML cache -> Markdown sạch + YAML frontmatter.

Vì sao tách khỏi fetch: parser còn phải sửa nhiều lần; có HTML cache thì
sửa xong parse lại chỉ mất vài giây, không phải cào lại toàn bộ.

Usage:
    python -m crawler.parse --canonical data/raw/canonical.jsonl \
        --manifest data/raw/manifest.jsonl --out data/processed/articles
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import yaml
from bs4 import BeautifulSoup, NavigableString, Tag

CUT_MARKERS_TEXT = [
    "đặt khám tiện lợi cùng youmed",
    "nguồn tham khảo",
    "chia sẻ thông tin hữu ích",
    "có thể bạn quan tâm",
    "tin tưởng nội dung từ chúng tôi",
    "tải ứng dụng youmed",
]
CUT_HEADING_MARKERS = [
    "nguồn tham khảo", "có thể bạn quan tâm", "tin tưởng nội dung", "chia sẻ thông tin",
]
JUNK_CLASS_HITS = (
    "ez-toc", "toc-", "yarpp", "related", "social-share", "share-post",
    "appointment", "banner", "advertisement", "sidebar", "adsbygoogle",
    "wp-block-buttons", "author-box", "post-author", "letter-section",
)


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def get_title(soup: BeautifulSoup) -> str:
    for sel in ("h1.entry-title", "article h1", "h1"):
        el = soup.select_one(sel)
        if el:
            return el.get_text(" ", strip=True)
    return ""


def get_content_root(soup: BeautifulSoup) -> Optional[Tag]:
    for sel in ("div.prose", "div.entry-content", "article .post-content", "article", "main"):
        el = soup.select_one(sel)
        if el:
            return el
    return None


def get_meta(soup: BeautifulSoup, prop: str) -> Optional[str]:
    el = soup.select_one(f'meta[property="{prop}"]')
    val = (el.get("content") or "").strip() if el else ""
    return val or None


def get_author(soup: BeautifulSoup) -> tuple[Optional[str], Optional[str]]:
    article = soup.select_one("article") or soup
    candidates = article.find_all("a", href=re.compile(r"/tin-tuc/bac-si/"))
    author_el = next((a for a in candidates if a.get_text(strip=True)), None)
    if author_el is None:
        return None, None
    author = author_el.get_text(strip=True)
    specialty = None
    sib = author_el.find_next_sibling("div")
    if sib:
        text = sib.get_text(" ", strip=True)
        if text.lower().startswith("chuyên khoa"):
            specialty = text.split(":", 1)[-1].strip()
    return author, specialty


# --------------------------------------------------------------------------- #
# Content extraction helpers
# --------------------------------------------------------------------------- #
def is_cut_marker(el: Tag) -> bool:
    txt = el.get_text(" ", strip=True).lower()
    if not txt:
        return False
    return any(m in txt and len(txt) < 200 for m in CUT_MARKERS_TEXT)


def is_junk_block(el: Tag) -> bool:
    classes = " ".join(el.get("class") or [])
    id_ = el.get("id") or ""
    if any(h in classes or h in id_ for h in JUNK_CLASS_HITS):
        return True
    if el.name in ("script", "style", "iframe"):
        return True
    if el.name == "figure" and el.find("img") and not el.get_text(strip=True):
        return True
    return False


def clean_ws(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()


def is_internal_article_link(href: str, canonical_url: str) -> bool:
    if "youmed.vn/tin-tuc/" not in href:
        return False
    if "/tin-tuc/bac-si/" in href:
        return False
    return href.rstrip("/") != canonical_url.rstrip("/")


def inline_md(el, canonical_url: str, related: List[str]) -> str:
    if isinstance(el, NavigableString):
        return str(el)
    if not isinstance(el, Tag):
        return ""
    if el.name in ("script", "style"):
        return ""
    if el.name == "br":
        return "\n"
    if el.name == "img":
        return ""

    inner = "".join(inline_md(c, canonical_url, related) for c in el.children)

    if el.name in ("strong", "b"):
        inner = inner.strip()
        return f"**{inner}**" if inner else ""
    if el.name in ("em", "i"):
        inner = inner.strip()
        return f"*{inner}*" if inner else ""
    if el.name == "a":
        # Chỉ giữ anchor text trong body; URL nội bộ đưa vào related_urls.
        href = (el.get("href") or "").strip()
        if href and is_internal_article_link(href, canonical_url) and href not in related:
            related.append(href)
        return inner.strip()
    if el.name == "code":
        return f"`{inner}`"
    if el.name in ("span", "u", "sub", "sup", "small", "font"):
        return inner
    return inner


def render_list(el: Tag, ordered: bool, canonical_url: str, related: List[str]) -> str:
    lines = []
    for i, li in enumerate(el.find_all("li", recursive=False), 1):
        content = clean_ws(inline_md(li, canonical_url, related))
        content = re.sub(r"\s+", " ", content.replace("\n", " ")).strip()
        if content:
            lines.append(f"{i}. {content}" if ordered else f"- {content}")
    return "\n".join(lines)


def render_table(el: Tag, canonical_url: str, related: List[str]) -> str:
    rows: List[List[str]] = []
    header: Optional[List[str]] = None
    for tr in el.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        row = [clean_ws(inline_md(c, canonical_url, related)).replace("\n", " ").replace("|", "\\|") for c in cells]
        if header is None and all(c.name == "th" for c in cells):
            header = row
        else:
            rows.append(row)
    if header is None and rows:
        header = rows.pop(0)
    if not header:
        return ""
    ncol = len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * ncol) + " |"]
    for r in rows:
        while len(r) < ncol:
            r.append("")
        lines.append("| " + " | ".join(r[:ncol]) + " |")
    return "\n".join(lines)


def render_block(el: Tag, canonical_url: str, related: List[str]) -> str:
    name = el.name
    if name in ("h2", "h3", "h4", "h5", "h6"):
        text = clean_ws(inline_md(el, canonical_url, related))
        return f"{'#' * int(name[1])} {text}" if text else ""
    if name == "p":
        return clean_ws(inline_md(el, canonical_url, related))
    if name == "ul":
        return render_list(el, False, canonical_url, related)
    if name == "ol":
        return render_list(el, True, canonical_url, related)
    if name == "blockquote":
        text = clean_ws(inline_md(el, canonical_url, related))
        return "\n".join(f"> {ln}" for ln in text.split("\n") if ln.strip())
    if name == "table":
        return render_table(el, canonical_url, related)
    if name in ("figure", "div", "section"):
        parts = []
        for c in el.children:
            if isinstance(c, Tag) and not is_junk_block(c):
                out = render_block(c, canonical_url, related)
                if out:
                    parts.append(out)
        return "\n\n".join(parts)
    return ""


def walk_content(root: Tag, canonical_url: str, related: List[str]) -> str:
    parts: List[str] = []
    for child in root.children:
        if isinstance(child, NavigableString):
            txt = str(child).strip()
            if txt:
                parts.append(clean_ws(txt))
            continue
        if not isinstance(child, Tag) or is_junk_block(child):
            continue
        if is_cut_marker(child):
            break
        if child.name in ("h2", "h3", "h4"):
            htxt = child.get_text(" ", strip=True).lower()
            if any(m in htxt for m in CUT_HEADING_MARKERS):
                break
        block = render_block(child, canonical_url, related)
        if block:
            parts.append(block)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Article pipeline
# --------------------------------------------------------------------------- #
def parse_article(html: str, canonical_url: str) -> Optional[dict]:
    soup = BeautifulSoup(html, "lxml")
    title = get_title(soup)
    root = get_content_root(soup)
    if not title or root is None:
        return None

    related: List[str] = []
    body = walk_content(root, canonical_url, related)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if not body:
        return None

    published = get_meta(soup, "article:published_time")
    updated = get_meta(soup, "article:modified_time")
    author, specialty = get_author(soup)

    return {
        "title": title,
        "body": body,
        "category": get_meta(soup, "article:section"),
        "author": author,
        "specialty": specialty,
        "published_date": published[:10] if published else None,
        "updated_date": updated[:10] if updated else None,
        "related_urls": related,
    }


def write_article_md(out_dir: str, group: dict, manifest_row: dict, parsed: dict) -> str:
    frontmatter = {
        "article_id": group["article_id"],
        "canonical_url": group["canonical_url"],
        "index_urls": group["index_urls"],
        "title": parsed["title"],
        "aliases": group["aliases"],
        "category": parsed["category"],
        "author": parsed["author"],
        "specialty": parsed["specialty"],
        "published_date": parsed["published_date"],
        "updated_date": parsed["updated_date"],
        "related_urls": parsed["related_urls"],
        "fetched_at": manifest_row.get("fetched_at"),
        "html_sha256": manifest_row.get("html_sha256"),
    }
    yaml_text = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{group['article_id']}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"---\n{yaml_text}---\n\n# {parsed['title']}\n\n{parsed['body']}\n")
    return out_path


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", default="data/raw/canonical.jsonl")
    ap.add_argument("--manifest", default="data/raw/manifest.jsonl")
    ap.add_argument("--out", default="data/processed/articles")
    args = ap.parse_args(argv)

    manifest_by_html_path: Dict[str, dict] = {}
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                manifest_by_html_path[row["html_path"]] = row

    ok = empty = err = 0
    total = 0
    with open(args.canonical, encoding="utf-8") as f:
        groups = [json.loads(line) for line in f if line.strip()]

    for group in groups:
        total += 1
        try:
            html = open(group["html_path"], encoding="utf-8").read()
            parsed = parse_article(html, group["canonical_url"])
            if parsed is None:
                empty += 1
                print(f"[EMPTY] {group['article_id']}", file=sys.stderr)
                continue
            manifest_row = manifest_by_html_path.get(group["html_path"], {})
            write_article_md(args.out, group, manifest_row, parsed)
            ok += 1
        except Exception as exc:
            err += 1
            print(f"[ERR] {group['article_id']}: {exc}", file=sys.stderr)

    print(f"\n[done] ok={ok} empty={empty} err={err} / {total}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
