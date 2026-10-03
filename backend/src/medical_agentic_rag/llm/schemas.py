"""Pydantic schema cho output có cấu trúc của từng task (mục 7.2, 7.3, 8.1, 8.2, 8.4)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class ChunkJudgement(BaseModel):
    n: int  # số thứ tự chunk trong context, khớp [n]
    relevant: bool
    covers: list[str]  # khía cạnh (chỉ chọn trong danh sách được đưa) mà chunk trả lời


class EvidenceGrade(BaseModel):
    judgements: list[ChunkJudgement]


class Rewrites(BaseModel):
    queries: list[str]  # 1-3 truy vấn, không trùng câu gốc
    rationale: str


class Condensed(BaseModel):
    standalone_question: str
    is_followup: bool
    carried_entities: list[str]  # thực thể kế thừa từ lượt trước


QuestionType = Literal[
    "definition", "section_lookup", "symptom_to_condition",
    "comparison", "multi_aspect", "complex", "out_of_scope",
]
SafetyLabel = Literal[
    "normal", "diagnosis_request", "medication_dosing",
    "high_risk", "emergency", "self_harm",
]


class QueryAnalysis(BaseModel):
    question_type: QuestionType
    safety_label: SafetyLabel
    entities: list[str]              # tên bệnh / triệu chứng / thuốc
    aspects: list[str]               # khía cạnh người hỏi cần, VD ["nguyên nhân", "khi nào đi khám"]
    target_section_types: list[str]  # map aspects sang section_type của Phase 1


class SubqueryPlan(BaseModel):
    subqueries: list[str]                    # tối đa 4
    section_types: list[Optional[str]]       # lọc tương ứng từng subquery
