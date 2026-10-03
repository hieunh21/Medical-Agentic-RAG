"""Prompt cho các task Phase 2 (grade_evidence, rewrite_query) và Phase 3
(condense_question, analyze_query, plan_subqueries)."""
from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

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


CONDENSE_TEMPLATE = """Dựa vào LỊCH SỬ hội thoại bên dưới, biến CÂU HỎI MỚI (có thể là câu nối tiếp,
dùng đại từ "nó", "bệnh đó"...) thành một câu hỏi độc lập, đầy đủ ý, hiểu được mà không cần đọc lịch sử.

Nếu CÂU HỎI MỚI đã độc lập (không phụ thuộc lịch sử), giữ nguyên nội dung, is_followup=false.

BÀI ĐANG ĐƯỢC NÓI TỚI: {active_titles}

LỊCH SỬ (tối đa 3 lượt gần nhất):
{history}

CÂU HỎI MỚI: {question}

Trả về đúng schema JSON yêu cầu."""


def _format_history(history: List[AnyMessage]) -> str:
    lines = []
    for m in history[-6:]:  # tối đa 3 lượt = 3 cặp người dùng/trợ lý
        if isinstance(m, HumanMessage):
            lines.append(f"Người dùng: {m.content}")
        elif isinstance(m, AIMessage):
            content = str(m.content)[:300]
            lines.append(f"Trợ lý: {content}")
    return "\n".join(lines) if lines else "(lượt đầu, chưa có lịch sử)"


def build_condense_prompt(history: List[AnyMessage], question: str, active_titles: List[str]) -> str:
    return CONDENSE_TEMPLATE.format(
        active_titles=", ".join(active_titles) or "(chưa có)",
        history=_format_history(history),
        question=question,
    )


ANALYZE_TEMPLATE = """Phân tích CÂU HỎI y khoa sau:

CÂU HỎI: {question}

- question_type: một trong "definition" (hỏi khái niệm), "section_lookup" (hỏi 1 khía cạnh của
  1 bệnh cụ thể đã nêu tên), "symptom_to_condition" (mô tả triệu chứng, không nêu tên bệnh),
  "comparison" (so sánh 2+ đối tượng), "multi_aspect" (hỏi nhiều khía cạnh của 1 bệnh cùng lúc),
  "complex" (cần nối thông tin nhiều bài, không chia sẵn được thành câu hỏi con),
  "out_of_scope" (không liên quan y khoa).
- safety_label: một trong "normal", "diagnosis_request" (xin chẩn đoán "tôi bị bệnh gì"),
  "medication_dosing" (hỏi liều thuốc cụ thể), "high_risk" (triệu chứng nguy hiểm),
  "emergency" (cấp cứu), "self_harm" (tự hại).
- entities: tên bệnh/triệu chứng/thuốc được nhắc tới.
- aspects: khía cạnh người hỏi cần (vd "nguyên nhân", "triệu chứng", "khi nào đi khám",
  "chẩn đoán", "điều trị", "phòng ngừa", "biến chứng"). Nếu câu hỏi đơn giản, 1 khía cạnh là đủ.
- target_section_types: map mỗi aspect sang section_type tương ứng (when_to_see_doctor, symptom,
  cause_risk, diagnosis, treatment, prevention, complication_prognosis, care, overview, other).

Trả về đúng schema JSON yêu cầu."""


def build_analyze_prompt(question: str) -> str:
    return ANALYZE_TEMPLATE.format(question=question)


PLAN_SUBQUERIES_TEMPLATE = """Câu hỏi sau cần tách thành tối đa 4 truy vấn con để tìm kiếm riêng từng
phần, mỗi truy vấn tập trung vào 1 thực thể hoặc 1 khía cạnh:

CÂU HỎI: {question}
THỰC THỂ: {entities}
KHÍA CẠNH: {aspects}

Với mỗi truy vấn con, nếu khía cạnh map được sang section_type (when_to_see_doctor, symptom,
cause_risk, diagnosis, treatment, prevention, complication_prognosis, care, overview, other) thì
điền vào section_types tương ứng vị trí, không map được thì để null.

Trả về đúng schema JSON yêu cầu."""


def build_plan_subqueries_prompt(question: str, entities: List[str], aspects: List[str]) -> str:
    return PLAN_SUBQUERIES_TEMPLATE.format(
        question=question, entities=", ".join(entities) or "(không rõ)",
        aspects=", ".join(aspects) or "(không rõ)",
    )
