"""Node: grade_evidence — LLM chấm từng chunk có/không, code tính coverage (mục 7.2)."""
from __future__ import annotations

from medical_agentic_rag.graph.state import State
from medical_agentic_rag.llm.client import run_task
from medical_agentic_rag.llm.prompts import build_grade_prompt
from medical_agentic_rag.llm.schemas import EvidenceGrade


async def grade_evidence(state: State) -> dict:
    chunks = state["reranked_chunks"]
    aspects = state["aspects"]

    if not chunks:
        return {"n_relevant": 0, "coverage": 0.0, "missing_aspects": list(aspects)}

    prompt = build_grade_prompt(state["standalone_question"], chunks, aspects)
    grade: EvidenceGrade = await run_task("grade_evidence", prompt, state, schema=EvidenceGrade)

    relevant = [j for j in grade.judgements if j.relevant]
    covered = {a for j in relevant for a in j.covers} & set(aspects)
    coverage = len(covered) / max(len(aspects), 1)
    missing = [a for a in aspects if a not in covered]

    return {
        "n_relevant": len(relevant), "coverage": coverage, "missing_aspects": missing,
        "llm_calls": state["llm_calls"],  # run_task() đã mutate state["llm_calls"] in-place
    }
