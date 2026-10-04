"""Safety lớp 1 (rule, không LLM) + hợp nhất với nhãn của lớp 2 (analyze_query) — mục 10.1.

Khớp theo ranh giới từ. Người dùng gõ có dấu thì khớp cụm có dấu (tránh "ngạt mũi" dính
"ngất"); gõ hoàn toàn không dấu thì khớp thêm trên bản bỏ dấu.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

KEYWORDS_PATH = Path(__file__).parent / "keywords_vi.yaml"

# Tăng dần theo mức nghiêm trọng; "lấy nhãn nghiêm trọng hơn" giữa hai lớp.
SEVERITY = ["normal", "diagnosis_request", "medication_dosing", "high_risk", "emergency", "self_harm"]
STOP_LABELS = frozenset({"emergency", "self_harm"})  # dừng pipeline, trả template


def strip_accents(s: str) -> str:
    s = s.lower().replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _compile(phrases: list[str]) -> re.Pattern:
    alts = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alts})(?!\w)")


@lru_cache(maxsize=1)
def _patterns() -> dict[str, tuple[re.Pattern, re.Pattern]]:
    data = yaml.safe_load(KEYWORDS_PATH.read_text(encoding="utf-8")) or {}
    out = {}
    for label, phrases in data.items():
        phrases = [unicodedata.normalize("NFC", p.lower()) for p in phrases]
        out[label] = (_compile(phrases), _compile([strip_accents(p) for p in phrases]))
    return out


def has_personal_marker(text: str) -> bool:
    """Câu có dấu hiệu người hỏi / người thân đang gặp tình huống cụ thể (tôi, con tôi, đang, vừa...).

    Câu gõ hoàn toàn không dấu chỉ so khớp bản bỏ dấu (nếu không, "em" của "trẻ em" lọt qua trước khi
    cụm chung chung bị loại); câu có dấu chỉ so khớp bản có dấu.
    """
    lowered = unicodedata.normalize("NFC", text.lower())
    unaccented = strip_accents(text)
    generic_accented, generic_plain = _patterns()["generic_phrases"]
    markers_accented, markers_plain = _patterns()["personal_markers"]
    if lowered.replace("đ", "d") == unaccented:
        return bool(markers_plain.search(generic_plain.sub(" ", unaccented)))
    return bool(markers_accented.search(generic_accented.sub(" ", lowered)))


def more_severe(a: Optional[str], b: Optional[str]) -> str:
    a, b = a or "normal", b or "normal"
    rank = {l: i for i, l in enumerate(SEVERITY)}
    return a if rank.get(a, 0) >= rank.get(b, 0) else b


def rule_label(text: str) -> str:
    """Nhãn lớp 1: "self_harm" | "emergency" | "normal"."""
    lowered = unicodedata.normalize("NFC", text.lower())
    unaccented = strip_accents(text)
    typed_without_accents = lowered.replace("đ", "d") == unaccented
    for label in ("self_harm", "emergency"):  # self_harm ưu tiên khi cả hai cùng khớp
        accented_re, plain_re = _patterns()[label]
        if accented_re.search(lowered) or (typed_without_accents and plain_re.search(unaccented)):
            return label
    return "normal"
