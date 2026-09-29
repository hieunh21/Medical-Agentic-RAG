"""Sparse vector cho tiếng Việt: unigram + bigram âm tiết + bản không dấu.

Tiếng Việt là ngôn ngữ âm tiết, một từ thường gồm 2-3 âm tiết, và người
dùng hay gõ không dấu. Value là phần TF-saturation của BM25 tính phía
client; IDF để Qdrant tự tính (Modifier.IDF).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter


def _strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def vi_terms(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    syl = re.findall(r"\w+", text)
    grams = syl + [f"{a}_{b}" for a, b in zip(syl, syl[1:])]
    return grams + ["~" + _strip_accents(g) for g in grams]


def _idx(term: str) -> int:
    return int.from_bytes(hashlib.blake2b(term.encode(), digest_size=4).digest(), "little")


def doc_sparse(text: str, avg_len: float, k1: float = 1.2, b: float = 0.75) -> tuple[list[int], list[float]]:
    tf = Counter(_idx(t) for t in vi_terms(text))
    n = sum(tf.values())
    norm = k1 * (1 - b + b * n / avg_len) if avg_len else k1
    return list(tf.keys()), [f * (k1 + 1) / (f + norm) for f in tf.values()]


def query_sparse(text: str) -> tuple[list[int], list[float]]:
    terms = vi_terms(text)
    if _strip_accents(text.lower()) == text.lower():
        terms = [t for t in terms if t.startswith("~")]
    idx = sorted({_idx(t) for t in terms})
    return idx, [1.0] * len(idx)


def doc_term_count(text: str) -> int:
    """Số term (trước hash) — dùng để tính avg_len của corpus."""
    return len(vi_terms(text))
