"""Bảng task: model theo tier (fast/strong/judge), temperature, max_output_tokens.

Đổi model chỉ cần sửa biến môi trường LLM_MODEL_FAST/STRONG/JUDGE, không sửa code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    model_tier: str  # "lite" | "fast" | "strong" | "judge"
    temperature: float
    max_output_tokens: int
    # 0 = tắt thinking hẳn; -1 = để model tự quyết mức thinking (dynamic).
    # Mặc định tắt (0): các task ở đây là trích xuất/sinh có cấu trúc, graph/code
    # đã lo phần rẽ nhánh — thinking chỉ tổ ăn vào max_output_tokens (từng gây
    # research_agent trả rỗng giữa vòng lặp, xem agent/loop.py). plan_subqueries
    # là ngoại lệ: tách câu hỏi thành subquery cần suy luận thật, tắt thinking làm
    # subquery kém đi (section_coverage giảm trên bộ M khi test).
    thinking_budget: int = 0


TASKS: dict[str, TaskSpec] = {
    "generate_answer": TaskSpec("strong", 0.2, 1024),
    # grade là phân loại có/không trên từng nguồn, không cần model mạnh — và nó gửi cả context
    # nên là task tốn token nhất mỗi câu. Đặt LLM_MODEL_LITE để nó chạy model rẻ hơn.
    "grade_evidence": TaskSpec("lite", 0.0, 1024),
    "rewrite_query": TaskSpec("fast", 0.3, 512),
    "condense_question": TaskSpec("fast", 0.0, 512),
    # 1024 (không phải 512) — JSON schema (entities/aspects/target_section_types) dễ
    # dài hơn tưởng, 512 từng bị cắt giữa chuỗi gây lỗi parse (xem run_routing_accuracy).
    "analyze_query": TaskSpec("fast", 0.0, 1024),
    "plan_subqueries": TaskSpec("fast", 0.0, 1024, thinking_budget=-1),
    "research_agent": TaskSpec("strong", 0.0, 1024),
    "verify_claims": TaskSpec("judge", 0.0, 2048),
}
