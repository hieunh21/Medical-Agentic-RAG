"""Node safety_rules — lớp 1 (rule, không LLM), chạy đầu tiên (mục 10.1).

emergency mà câu không có dấu hiệu tình huống cá nhân (vd. "Chứng co giật nửa mặt có nguy hiểm không?")
được hoãn cho lớp 2 (analyze_query) quyết định là hỏi kiến thức chung hay tình huống thật, thay vì dừng
ngay. self_harm và mọi emergency có dấu hiệu cá nhân vẫn dừng ngay, không chờ lớp 2.
"""
from __future__ import annotations

from medical_agentic_rag.graph.state import State
from medical_agentic_rag.safety.rules import has_personal_marker, more_severe, rule_label


def safety_rules(state: State) -> dict:
    label = rule_label(state["question"])
    deferred = label == "emergency" and not has_personal_marker(state["question"])
    return {
        "safety_label": more_severe(state.get("safety_label"), label),
        "safety_deferred": deferred,
    }
