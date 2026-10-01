"""Prompt cho các task Phase 2: grade_evidence, rewrite_query."""
from __future__ import annotations

from typing import List

from medical_agentic_rag.answer import format_context

GRADE_TEMPLATE = """Bạn chấm điểm evidence cho hệ thống hỏi đáp y khoa.
Với MỖI chunk trong CONTEXT, xác định:
- relevant: chunk có chứa thông tin giúp trả lời CÂU HỎI không.
- covers: chunk trả lời được khía cạnh nào trong KHÍA CẠNH dưới đây (chỉ chọn trong danh sách, có thể để trống).

KHÍA CẠNH: {aspects}

CÂU HỎI: {question}

CONTEXT:
{context}

Trả về đúng schema JSON; trường n phải khớp số thứ tự [n] của từng chunk trong CONTEXT."""


def build_grade_prompt(question: str, chunks: List[dict], aspects: List[str]) -> str:
    return GRADE_TEMPLATE.format(
        aspects=", ".join(aspects), question=question, context=format_context(chunks),
    )


REWRITE_TEMPLATE = """Câu hỏi sau không tìm được tài liệu liên quan trong YouMed: "{question}"

Hãy sinh tối đa 3 câu truy vấn thay thế theo các chiến lược:
1. Khôi phục dấu nếu câu hỏi gõ không dấu.
2. Chuyển văn nói / mô tả triệu chứng sang thuật ngữ y khoa.
3. Thử tên gọi khác của bệnh/triệu chứng (tên dân gian ↔ tên y khoa).

Không lặp lại câu hỏi gốc. Trả về đúng schema JSON yêu cầu."""


def build_rewrite_prompt(question: str) -> str:
    return REWRITE_TEMPLATE.format(question=question)
