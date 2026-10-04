"""Citation validator — phần code thuần (mục 10.2). Node gọi LLM nằm ở graph/nodes/validate.py.

Bước 1 (code): xoá [n] không có trong context; câu có nội dung y khoa mà không có [n] -> unsupported.
Bước 3 (chính sách): xoá câu unsupported; xoá quá nhiều thì yêu cầu viết lại.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

REF_RE = re.compile(r"\[(\d+)\]")
DOSE_RE = re.compile(r"\d+(?:[.,]\d+)?\s?(?:mg|ml|mcg|µg|viên)\b", re.IGNORECASE)
# Câu khuyến cáo / từ chối chung — không phải khẳng định y khoa cần nguồn.
NO_CITE_RE = re.compile(
    r"chưa có thông tin|chưa có bài|không đủ thông tin|nên (?:hỏi ý kiến|đi khám|gặp bác sĩ|liên hệ)|"
    r"hãy (?:hỏi|đi khám|tham khảo)|tham khảo ý kiến|không thể chẩn đoán|không phải là chẩn đoán",
    re.IGNORECASE,
)
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
MAX_REMOVED_RATIO = 0.30


@dataclass
class Sentence:
    id: int
    line: int
    text: str
    refs: list[int] = field(default_factory=list)
    needs_cite: bool = True
    verdict: str = ""        # supported | partial | unsupported (rỗng = chưa xét)
    reason: str = ""


def _needs_citation(text: str) -> bool:
    body = REF_RE.sub("", text).strip(" -*•\t")
    if len(re.findall(r"\w+", body)) < 4:
        return False
    if body.endswith(":"):  # câu dẫn
        return False
    return not NO_CITE_RE.search(body)


def split_sentences(text: str) -> list[Sentence]:
    out: list[Sentence] = []
    for li, line in enumerate(text.split("\n")):
        if not line.strip():
            continue
        for part in SENTENCE_END_RE.split(line.strip()):
            if part.strip():
                out.append(Sentence(id=len(out), line=li, text=part.strip()))
    return out


def check_citations(sentences: list[Sentence], n_chunks: int, restrict_dose: bool = False) -> None:
    """Bước 1: sửa tại chỗ. [n] lạ bị xoá; câu thiếu nguồn / chứa liều cá nhân bị đánh dấu unsupported."""
    for s in sentences:
        valid = lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_chunks else ""  # noqa: E731
        s.text = re.sub(r"[ \t]{2,}", " ", REF_RE.sub(valid, s.text)).strip()
        s.refs = sorted({int(n) for n in REF_RE.findall(s.text)})
        s.needs_cite = _needs_citation(s.text)
        if s.needs_cite and not s.refs:
            s.verdict, s.reason = "unsupported", "không có trích dẫn"
        elif restrict_dose and DOSE_RE.search(REF_RE.sub("", s.text)):
            s.verdict, s.reason = "unsupported", "chứa liều thuốc cụ thể"


def to_verify(sentences: list[Sentence]) -> list[Sentence]:
    """Câu cần LLM kiểm tra: có nội dung y khoa, có [n] hợp lệ, chưa bị loại ở bước 1."""
    return [s for s in sentences if s.needs_cite and s.refs and not s.verdict]


def apply_verdicts(sentences: list[Sentence], verdicts: dict[int, tuple[str, str]]) -> None:
    for s in sentences:
        if s.id in verdicts and not s.verdict:
            s.verdict, s.reason = verdicts[s.id]


def claim_counts(sentences: list[Sentence]) -> dict[str, int]:
    counts = {"supported": 0, "partial": 0, "unsupported": 0}
    for s in sentences:
        if s.verdict in counts:
            counts[s.verdict] += 1
    return counts


def split_kept_removed(sentences: list[Sentence]) -> tuple[list[Sentence], list[Sentence]]:
    kept = [s for s in sentences if s.verdict != "unsupported"]
    return kept, [s for s in sentences if s.verdict == "unsupported"]


def needs_regeneration(sentences: list[Sentence]) -> bool:
    """Bước 3: xoá > 30% số câu có nội dung y khoa, hoặc không còn câu nào."""
    kept, removed = split_kept_removed(sentences)
    if not removed:
        return False
    if not kept:
        return True
    claims = [s for s in sentences if s.needs_cite]
    return bool(claims) and len([s for s in removed if s.needs_cite]) / len(claims) > MAX_REMOVED_RATIO


def rebuild(sentences: list[Sentence]) -> str:
    lines: dict[int, list[str]] = {}
    for s in sentences:
        lines.setdefault(s.line, []).append(s.text)
    return "\n".join(" ".join(parts) for _, parts in sorted(lines.items()))
