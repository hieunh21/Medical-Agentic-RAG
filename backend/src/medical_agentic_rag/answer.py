"""Baseline generate_answer (Phase 1): context đánh số [n], bắt buộc trích dẫn."""
from __future__ import annotations

from typing import List

PROMPT_TEMPLATE = """Bạn là trợ lý thông tin y khoa. Chỉ dùng thông tin trong CONTEXT.
- Mỗi câu có nội dung y khoa phải có trích dẫn dạng [n].
- Không chẩn đoán cho người hỏi, không đưa liều thuốc cá nhân.
- Nếu CONTEXT không đủ để trả lời, nói rõ YouMed chưa có thông tin.

CONTEXT:
{context}

CÂU HỎI: {question}"""


def format_context(chunks: List[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        updated = f" — cập nhật {c['updated_date']}" if c.get("updated_date") else ""
        lines.append(f"[{i}] ({c['title']}{updated}) {c['text']}")
    return "\n".join(lines)


def build_prompt(question: str, chunks: List[dict]) -> str:
    return PROMPT_TEMPLATE.format(context=format_context(chunks), question=question)


def format_sources(chunks: List[dict]) -> List[dict]:
    return [
        {"n": i, "title": c["title"], "url": c["url"], "updated_date": c.get("updated_date")}
        for i, c in enumerate(chunks, 1)
    ]
