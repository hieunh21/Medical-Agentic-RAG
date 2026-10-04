"""Node analyze_query — phân loại câu hỏi, tách thực thể/khía cạnh (mục 8.2).

safety_label được thu thập ở đây nhưng hành vi xử lý thật sự (dừng pipeline,
template cấp cứu...) để Phase 5 — xem mục 10.1.
"""
from __future__ import annotations

from medical_agentic_rag.budget import degrade_on_budget
from medical_agentic_rag.graph.state import DEFAULT_ASPECT, State
from medical_agentic_rag.llm.client import run_task
from medical_agentic_rag.llm.prompts import build_analyze_prompt
from medical_agentic_rag.llm.schemas import QueryAnalysis


# Không phân loại được -> question_type rỗng -> route_by_type rơi về hybrid_retrieve.
@degrade_on_budget(lambda s: {"question_type": "", "aspects": [DEFAULT_ASPECT]})
async def analyze_query(state: State) -> dict:
    prompt = build_analyze_prompt(state["standalone_question"])
    analysis: QueryAnalysis = await run_task("analyze_query", prompt, state, schema=QueryAnalysis)
    return {
        "question_type": analysis.question_type,
        "safety_label": analysis.safety_label,
        "entities": analysis.entities,
        "aspects": analysis.aspects or [DEFAULT_ASPECT],
        "target_section_types": analysis.target_section_types,
        "llm_calls": state["llm_calls"],  # run_task() đã mutate state["llm_calls"] in-place
    }
