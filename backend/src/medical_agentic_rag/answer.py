"""generate_answer: context đánh số [n], bắt buộc trích dẫn (Phase 1 + 2)."""
from __future__ import annotations

from typing import List, Optional

PROMPT_TEMPLATE = """Bạn là trợ lý thông tin y khoa. Chỉ dùng thông tin trong CONTEXT.
- Mỗi câu có nội dung y khoa phải có trích dẫn dạng [n].
- Không chẩn đoán cho người hỏi, không đưa liều thuốc cá nhân.
- Nếu CONTEXT không đủ để trả lời, nói rõ YouMed chưa có thông tin.
{partial_note}
CONTEXT:
{context}

CÂU HỎI: {question}"""

PARTIAL_NOTE = "- CONTEXT chỉ trả lời được một phần câu hỏi, hãy nói rõ phần nào chưa có thông tin.\n"
SYMPTOM_NOTE = (
    "- Người hỏi mô tả triệu chứng, không nêu tên bệnh: chỉ nêu các khả năng được CONTEXT "
    "mô tả, không kết luận người hỏi mắc bệnh gì, luôn khuyên đi khám để chẩn đoán chính xác.\n"
)


DIAGNOSIS_NOTE = (
    "- Người hỏi xin chẩn đoán: chỉ nêu các khả năng được CONTEXT mô tả, không kết luận người hỏi "
    "mắc bệnh gì, khuyên đi khám để được chẩn đoán.\n"
)
DOSING_NOTE = (
    "- Câu hỏi liên quan liều thuốc: chỉ nêu thông tin chung về thuốc nếu CONTEXT có; TUYỆT ĐỐI không "
    "đưa liều dùng (mg/ml/viên) cho người hỏi; khuyên hỏi bác sĩ hoặc dược sĩ.\n"
)
HIGH_RISK_NOTE = (
    "- Triệu chứng có thể nguy hiểm: nhấn mạnh nên đi khám sớm và nêu dấu hiệu cần cấp cứu nếu CONTEXT có.\n"
)
REGEN_NOTE = (
    "- Bản nháp trước bị loại các câu sau vì không được CONTEXT hỗ trợ hoặc vi phạm quy tắc. KHÔNG lặp "
    "lại các ý này; chỉ viết điều CONTEXT nói rõ và gắn [n] đúng nguồn:\n{rejected}\n"
)
SAFETY_NOTES = {
    "diagnosis_request": DIAGNOSIS_NOTE, "medication_dosing": DOSING_NOTE, "high_risk": HIGH_RISK_NOTE,
}


def format_context(chunks: List[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        updated = f" — cập nhật {c['updated_date']}" if c.get("updated_date") else ""
        lines.append(f"[{i}] ({c['title']}{updated}) {c['text']}")
    return "\n".join(lines)


def build_prompt(
    question: str, chunks: List[dict],
    evidence_status: Optional[str] = None, question_type: Optional[str] = None,
    safety_label: Optional[str] = None, rejected: Optional[List[str]] = None,
) -> str:
    notes = ""
    if evidence_status == "partial":
        notes += PARTIAL_NOTE
    if question_type == "symptom_to_condition":
        notes += SYMPTOM_NOTE
    notes += SAFETY_NOTES.get(safety_label or "", "")
    if rejected:
        notes += REGEN_NOTE.format(rejected="\n".join(f"  + {r}" for r in rejected))
    return PROMPT_TEMPLATE.format(context=format_context(chunks), question=question, partial_note=notes)


def format_sources(chunks: List[dict]) -> List[dict]:
    return [
        {"n": i, "title": c["title"], "url": c["url"], "updated_date": c.get("updated_date")}
        for i, c in enumerate(chunks, 1)
    ]
