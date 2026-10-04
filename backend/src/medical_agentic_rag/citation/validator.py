"""Citation validator — phần code thuần (mục 10.2). Node gọi LLM nằm ở graph/nodes/validate.py.

Bước 1 (code): xoá [n] không có trong context; câu có nội dung y khoa mà không có [n] -> unsupported.
Bước 3 (chính sách): xoá câu unsupported; xoá quá nhiều thì yêu cầu viết lại.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Một nhóm trích dẫn: "[1]" và cả "[1, 3]" / "[1,3]" — model hay gộp nhiều nguồn vào một cặp
# ngoặc dù prompt yêu cầu dạng [n] (gặp ở ~7% câu trả lời). Chỉ khớp "[1]" thì các câu đó bị coi
# là KHÔNG có trích dẫn -> unsupported -> validator xoá sạch một câu trả lời vốn đúng.
REF_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
REF_NUM_RE = re.compile(r"\d+")
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
    def keep_valid(m: re.Match) -> str:
        """Giữ các số có thật trong context, viết lại thành [a][b]; không còn số nào thì xoá nhóm."""
        nums = [int(x) for x in REF_NUM_RE.findall(m.group(0))]
        return "".join(f"[{x}]" for x in dict.fromkeys(n for n in nums if 1 <= n <= n_chunks))

    for s in sentences:
        s.text = re.sub(r"[ \t]{2,}", " ", REF_RE.sub(keep_valid, s.text)).strip()
        s.refs = sorted({int(x) for g in REF_RE.findall(s.text) for x in REF_NUM_RE.findall(g)})
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


def _remap_group(match: re.Match, mapping: dict[int, int]) -> str:
    nums = [mapping[int(x)] for x in REF_NUM_RE.findall(match.group(0)) if int(x) in mapping]
    return "".join(f"[{x}]" for x in dict.fromkeys(nums))


def renumber_citations(sentences: list[Sentence], sources: list[dict]) -> list[dict]:
    """Bỏ nguồn không câu nào trích, đánh số lại 1..K và sửa [n] trong các câu được giữ.

    Retrieval lấy tới FINAL_CONTEXT_K nguồn nhưng câu trả lời thường chỉ trích vài cái; hiện cả
    nguồn không được trích làm người đọc tưởng câu trả lời dựa trên chúng — có khi lệch hẳn chủ
    đề (vd. câu hỏi bong gân mà danh sách nguồn có bài về một bệnh lây qua đường tình dục).
    Sửa tại chỗ `sentences`, trả về danh sách nguồn mới.
    """
    cited = sorted({n for s in sentences for n in s.refs if 1 <= n <= len(sources)})
    mapping = {old: new for new, old in enumerate(cited, 1)}
    for s in sentences:
        s.text = REF_RE.sub(lambda m: _remap_group(m, mapping), s.text)
        s.refs = sorted({mapping[n] for n in s.refs if n in mapping})
    return [{**sources[old - 1], "n": new} for old, new in mapping.items()]


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
