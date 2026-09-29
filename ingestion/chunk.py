#!/usr/bin/env python3
"""Chunk: Markdown YouMed -> chunks.jsonl, section-aware theo H2/H3.

Quy tắc (thiết kế mục 6.4):
  1. Ranh giới chính là H2; H3 nằm trong chunk của H2 cha.
  2. H2 dài hơn ~400 token -> tách tại H3; vẫn dài -> tách theo đoạn, overlap 1 đoạn.
  3. H3 kết thúc bằng ":" không phải ranh giới -> gắn liền với nội dung sau nó.
  4. Đoạn in đậm trước H2 đầu tiên -> chunk riêng, section_type=overview.
  5. Bảng -> chunk riêng (chunk_type=table).
  6. Section ngắn hơn ~60 token -> gộp vào chunk liền kề cùng H2 cha.
  7. Dedupe chunk theo hash của text đã chuẩn hoá.

Usage:
    python -m ingestion.chunk --articles data/processed/articles --out data/processed/chunks.jsonl
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from typing import List, Optional, TypedDict

import yaml

from ingestion.section_types import classify_section_type

MAX_TOKENS = 400
MIN_TOKENS = 60


class Chunk(TypedDict):
    chunk_id: str
    article_id: str
    url: str
    title: str
    aliases: list
    category: Optional[str]
    section_path: list
    section_type: str
    chunk_type: str
    chunk_index: int
    text: str
    n_tokens: int
    published_date: Optional[str]
    updated_date: Optional[str]
    content_hash: str
    table_markdown: Optional[str]


def approx_tokens(text: str) -> int:
    """Đếm từ xấp xỉ (không phải tokenizer thật của EmbeddingGemma) — đủ để so ngưỡng."""
    return len(re.findall(r"\w+", text, re.UNICODE))


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Parse markdown body thành block (dựa trên "\n\n" — đúng cách parse.py ghi ra)
# --------------------------------------------------------------------------- #
def block_kind(block: str) -> str:
    first_line = block.lstrip().splitlines()[0]
    if first_line.startswith("## "):
        return "h2"
    if first_line.startswith("### "):
        return "h3"
    if re.match(r"^#{4,6} ", first_line):
        return "h4plus"
    if first_line.startswith("|") and "---" in block:
        return "table"
    return "text"


def heading_text(block: str) -> str:
    return re.sub(r"^#+\s*", "", block.lstrip().splitlines()[0]).strip()


def h4plus_to_text(block: str) -> str:
    lines = block.splitlines()
    lines[0] = f"**{heading_text(lines[0])}**"
    return "\n".join(lines)


def split_body_into_h2(body: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Trả về (intro_blocks, [(h2_heading, blocks)])."""
    blocks = [b.strip("\n") for b in body.split("\n\n") if b.strip()]
    intro: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    current_heading: Optional[str] = None
    current_blocks: list[str] = []
    for block in blocks:
        if block_kind(block) == "h2":
            if current_heading is not None:
                sections.append((current_heading, current_blocks))
            current_heading = heading_text(block)
            current_blocks = []
        elif current_heading is None:
            intro.append(block)
        else:
            current_blocks.append(block)
    if current_heading is not None:
        sections.append((current_heading, current_blocks))
    return intro, sections


def extract_tables(blocks: list[str]) -> tuple[list[str], list[str]]:
    """Tách block bảng ra khỏi list block text."""
    text_blocks, tables = [], []
    for b in blocks:
        (tables if block_kind(b) == "table" else text_blocks).append(b)
    return text_blocks, tables


def split_into_h3_segments(blocks: list[str]) -> list[dict]:
    """Chia theo H3; H4+ gộp vào text; H3 kết ':' không tạo boundary mới."""
    segments: list[dict] = [{"heading": None, "blocks": []}]
    for block in blocks:
        kind = block_kind(block)
        if kind == "h3":
            h3_text = heading_text(block)
            starts_new = not h3_text.rstrip().endswith(":")
            if starts_new and (segments[-1]["blocks"] or segments[-1]["heading"]):
                segments.append({"heading": h3_text, "blocks": []})
            elif segments[-1]["heading"] is None and not segments[-1]["blocks"]:
                segments[-1]["heading"] = h3_text
            # else: gắn liền — không đổi heading, chỉ thêm block bên dưới
        elif kind == "h4plus":
            segments[-1]["blocks"].append(h4plus_to_text(block))
        else:
            segments[-1]["blocks"].append(block)
    return [s for s in segments if s["blocks"] or s["heading"]]


def segment_text(seg: dict) -> str:
    return "\n\n".join(seg["blocks"])


def split_paragraphs_overlap(text: str, max_tokens: int) -> list[str]:
    """Tách theo đoạn khi 1 segment vẫn quá dài, overlap 1 đoạn giữa các phần.

    Đơn vị tách là đoạn ("\n\n"); một đoạn/list tự nó đã quá dài (không có
    "\n\n" bên trong, ví dụ 1 list block dài) thì tách tiếp theo dòng.
    """
    if approx_tokens(text) <= max_tokens:
        return [text]

    units = [p for p in text.split("\n\n") if p.strip()]
    if len(units) <= 1:
        units = [ln for ln in text.split("\n") if ln.strip()]
    if len(units) <= 1:
        return [text]  # không còn ranh giới an toàn để tách tiếp

    # Unit nào tự nó đã vượt ngưỡng (vd: 1 list block dài) -> tách theo dòng.
    expanded: list[str] = []
    for u in units:
        if approx_tokens(u) > max_tokens:
            lines = [ln for ln in u.split("\n") if ln.strip()]
            expanded.extend(lines if len(lines) > 1 else [u])
        else:
            expanded.append(u)
    units = expanded

    out: list[str] = []
    current: list[str] = []
    for u in units:
        current.append(u)
        if approx_tokens("\n".join(current)) >= max_tokens:
            out.append("\n".join(current))
            current = [u]  # overlap: đơn vị cuối làm đơn vị mở đầu phần sau
    if current and (len(current) > 1 or not out):
        out.append("\n".join(current))
    return out or [text]


def _strip_md_emphasis(s: str) -> str:
    return re.sub(r"\*\*(.*?)\*\*|\*(.*?)\*", lambda m: m.group(1) or m.group(2), s)


def table_to_sentences(markdown_table: str) -> str:
    """Viết lại mỗi hàng bảng thành câu 'Cột A: …; Cột B: …' để embedding hiểu được."""
    rows = [r.strip() for r in markdown_table.splitlines() if r.strip()]
    if len(rows) < 2:
        return markdown_table
    header = [_strip_md_emphasis(c.strip()) for c in rows[0].strip("|").split("|")]
    sentences = []
    for row in rows[2:]:  # rows[1] là dòng "---"
        cells = [_strip_md_emphasis(c.strip()) for c in row.strip("|").split("|")]
        pairs = [f"{h}: {c}" for h, c in zip(header, cells) if c]
        if pairs:
            sentences.append("; ".join(pairs) + ".")
    return "\n".join(sentences)


def merge_short_chunks(items: list[dict]) -> list[dict]:
    """Rule 6: chunk ngắn hơn ~60 token gộp vào chunk liền kề cùng H2 cha."""
    if not items:
        return items
    merged = [dict(items[0])]
    for item in items[1:]:
        if approx_tokens(item["text"]) < MIN_TOKENS or approx_tokens(merged[-1]["text"]) < MIN_TOKENS:
            merged[-1]["text"] = merged[-1]["text"] + "\n\n" + item["text"]
        else:
            merged.append(dict(item))
    return merged


def chunk_h2_section(h2_heading: str, blocks: list[str]) -> list[dict]:
    """Trả về list {section_path, text} cho một H2 (chưa gán chunk_type/section_type)."""
    text_blocks, tables = extract_tables(blocks)
    full_text = "\n\n".join(text_blocks)
    results: list[dict] = []

    if approx_tokens(full_text) <= MAX_TOKENS:
        if full_text.strip():
            results.append({"section_path": [h2_heading], "text": full_text})
    else:
        segments = split_into_h3_segments(text_blocks)
        for seg in segments:
            path = [h2_heading, seg["heading"]] if seg["heading"] else [h2_heading]
            seg_txt = segment_text(seg)
            for part in split_paragraphs_overlap(seg_txt, MAX_TOKENS):
                if part.strip():
                    results.append({"section_path": path, "text": part})

    text_chunks = merge_short_chunks(results)

    table_chunks = [
        {"section_path": [h2_heading], "text": table_to_sentences(t), "table_markdown": t, "chunk_type": "table"}
        for t in tables
    ]
    return [{**c, "chunk_type": c.get("chunk_type", "text")} for c in text_chunks + table_chunks]


def chunk_article(md_path: str) -> List[Chunk]:
    raw = open(md_path, encoding="utf-8").read()
    _, fm_text, body = raw.split("---", 2)
    fm = yaml.safe_load(fm_text) or {}
    # bỏ dòng "# Title" đầu body
    body = re.sub(r"^\s*#\s+.*\n+", "", body, count=1)

    intro_blocks, h2_sections = split_body_into_h2(body)

    raw_chunks: list[dict] = []
    intro_text_blocks, intro_tables = extract_tables(intro_blocks)
    if "".join(intro_text_blocks).strip():
        raw_chunks.append({
            "section_path": [], "text": "\n\n".join(intro_text_blocks),
            "chunk_type": "text", "section_type": "overview",
        })
    for t in intro_tables:
        raw_chunks.append({
            "section_path": [], "text": table_to_sentences(t), "table_markdown": t,
            "chunk_type": "table", "section_type": "overview",
        })

    for h2_heading, blocks in h2_sections:
        section_type = classify_section_type(h2_heading)
        for c in chunk_h2_section(h2_heading, blocks):
            c["section_type"] = section_type
            raw_chunks.append(c)

    chunks: List[Chunk] = []
    for i, c in enumerate(raw_chunks):
        text = c["text"].strip()
        if not text:
            continue
        chunks.append({
            "chunk_id": f"{fm['article_id']}#{i:02d}",
            "article_id": fm["article_id"],
            "url": fm["canonical_url"],
            "title": fm["title"],
            "aliases": fm.get("aliases") or [],
            "category": fm.get("category"),
            "section_path": c["section_path"],
            "section_type": c["section_type"],
            "chunk_type": c["chunk_type"],
            "chunk_index": i,
            "text": text,
            "n_tokens": approx_tokens(text),
            "published_date": fm.get("published_date"),
            "updated_date": fm.get("updated_date"),
            "content_hash": content_hash(text),
            "table_markdown": c.get("table_markdown"),
        })
    return chunks


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", default="data/processed/articles")
    ap.add_argument("--out", default="data/processed/chunks.jsonl")
    args = ap.parse_args(argv)

    seen_hashes: set[str] = set()
    n_articles = n_chunks = n_dupes = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for path in sorted(glob.glob(f"{args.articles}/*.md")):
            n_articles += 1
            try:
                for chunk in chunk_article(path):
                    if chunk["content_hash"] in seen_hashes:
                        n_dupes += 1
                        continue
                    seen_hashes.add(chunk["content_hash"])
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    n_chunks += 1
            except Exception as exc:
                print(f"[ERR] {path}: {exc}", file=sys.stderr)

    print(f"[done] {n_articles} articles -> {n_chunks} chunks ({n_dupes} dedup) -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
