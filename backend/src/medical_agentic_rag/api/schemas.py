"""Schema request/response của API (mục 16). Tách khỏi schema LLM ở llm/schemas.py."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = None


class Source(BaseModel):
    n: int
    title: str
    url: str
    updated_date: Optional[str] = None


class Meta(BaseModel):
    """Số liệu để UI hiển thị panel "hệ thống đã trả lời thế nào"."""
    question_type: Optional[str] = None
    standalone_question: Optional[str] = None
    is_followup: bool = False
    safety_label: Optional[str] = None
    evidence_status: Optional[str] = None
    coverage: Optional[float] = None
    n_relevant: Optional[int] = None
    n_chunks: int = 0
    n_articles: int = 0
    corrections: int = 0
    regenerations: int = 0
    agent_tool_calls: int = 0
    llm_calls: int = 0
    claims: dict[str, int] = Field(default_factory=dict)
    elapsed_ms: float = 0.0


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    meta: Meta


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionHistory(BaseModel):
    session_id: str
    turns: list[Turn] = Field(default_factory=list)
    active_articles: list[str] = Field(default_factory=list)


class ServiceHealth(BaseModel):
    name: str
    ok: bool
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    ok: bool
    services: list[ServiceHealth]
