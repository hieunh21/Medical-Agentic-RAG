"""Pydantic schema cho output có cấu trúc của từng task (mục 7.2, 7.3)."""
from __future__ import annotations

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
