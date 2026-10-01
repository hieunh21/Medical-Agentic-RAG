"""Bảng task: model theo tier (fast/strong/judge), temperature, max_output_tokens.

Đổi model chỉ cần sửa biến môi trường LLM_MODEL_FAST/STRONG/JUDGE, không sửa code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    model_tier: str  # "fast" | "strong" | "judge"
    temperature: float
    max_output_tokens: int


TASKS: dict[str, TaskSpec] = {
    "generate_answer": TaskSpec("strong", 0.2, 1024),
    "grade_evidence": TaskSpec("fast", 0.0, 1024),
    "rewrite_query": TaskSpec("fast", 0.3, 256),
}
