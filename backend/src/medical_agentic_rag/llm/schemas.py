"""Pydantic schema cho output có cấu trúc của từng task (mục 7.2, 7.3, 8.1, 8.2, 8.4)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, field_validator


def as_str_list(value):
    """Model đôi khi trả object {"khía cạnh": "section_type"} hoặc 1 chuỗi thay vì danh sách (gây ~35% lỗi
    parse của analyze_query). Object -> danh sách các giá trị theo thứ tự; chuỗi -> danh sách 1 phần tử."""
    if isinstance(value, dict):
        value = list(value.values())
    elif isinstance(value, str):
        value = [value]
    if isinstance(value, list):
        return [x if isinstance(x, str) else str(x) for x in value if x is not None]
    return value


class ChunkJudgement(BaseModel):
    n: int  # số thứ tự chunk trong context, khớp [n]
    relevant: bool
    covers: list[str]  # khía cạnh (chỉ chọn trong danh sách được đưa) mà chunk trả lời

    _coerce_covers = field_validator("covers", mode="before")(as_str_list)


class EvidenceGrade(BaseModel):
    judgements: list[ChunkJudgement]


class Rewrites(BaseModel):
    queries: list[str]  # 1-3 truy vấn, không trùng câu gốc
    rationale: str

    _coerce_queries = field_validator("queries", mode="before")(as_str_list)


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
    situation: Literal["personal", "general"] = "personal"  # mơ hồ -> personal (an toàn hơn)
    entities: list[str]              # tên bệnh / triệu chứng / thuốc
    aspects: list[str]               # khía cạnh người hỏi cần, VD ["nguyên nhân", "khi nào đi khám"]
    target_section_types: list[str]  # map aspects sang section_type của Phase 1

    _coerce_lists = field_validator("entities", "aspects", "target_section_types", mode="before")(as_str_list)


class ClaimVerdict(BaseModel):
    sentence_id: int
    verdict: Literal["supported", "partial", "unsupported"]
    reason: str


class ClaimVerdicts(BaseModel):
    verdicts: list[ClaimVerdict]


class SubqueryPlan(BaseModel):
    subqueries: list[str]                    # tối đa 4
    section_types: list[Optional[str]]       # lọc tương ứng từng subquery
