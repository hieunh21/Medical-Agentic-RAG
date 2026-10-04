"""Node citation_validator — kiểm tra câu <-> nguồn (mục 10.2), chốt final_answer.

Bước 1 (code) và bước 3 (chính sách) ở citation/validator.py; bước 2 (verify_claims) là
1 lời gọi LLM gộp mọi câu. Lỗi LLM / hết ngân sách ở bước 2: bỏ qua bước 2, vẫn áp bước 1.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from medical_agentic_rag.budget import BudgetExceeded
from medical_agentic_rag.citation import validator as v
from medical_agentic_rag.errors import LLMTaskFailed
from medical_agentic_rag.graph.nodes.generate import NO_INFO
from medical_agentic_rag.graph.state import State
from medical_agentic_rag.llm.client import run_task
from medical_agentic_rag.llm.prompts import build_verify_claims_prompt
from medical_agentic_rag.llm.schemas import ClaimVerdicts
from medical_agentic_rag.safety.templates import EMERGENCY_TOPIC_NOTE, HIGH_RISK_PREFIX

FILTERED_NOTE = "\n\n(Lưu ý: một số thông tin chưa đủ nguồn nên đã được lược bỏ.)"


async def _verify(state: State, sentences: list[v.Sentence]) -> None:
    targets = v.to_verify(sentences)
    if not targets:
        return
    prompt = build_verify_claims_prompt([(s.id, s.text, s.refs) for s in targets], state["reranked_chunks"])
    try:
        result: ClaimVerdicts = await run_task("verify_claims", prompt, state, schema=ClaimVerdicts)
    except (BudgetExceeded, LLMTaskFailed):
        return
    v.apply_verdicts(sentences, {c.sentence_id: (c.verdict, c.reason) for c in result.verdicts})


def _finalize(state: State, text: str, extra: dict) -> dict:
    if text != NO_INFO:
        if state.get("safety_label") == "high_risk":
            text = HIGH_RISK_PREFIX + text
        elif state.get("emergency_topic") or state.get("safety_unverified"):
            text = EMERGENCY_TOPIC_NOTE + text
    return {
        "final_answer": text, "regenerate": False, "messages": [AIMessage(content=text)],
        "llm_calls": state["llm_calls"], **extra,
    }


async def validate_citations(state: State) -> dict:
    draft = state["draft_answer"]
    if state.get("generate_failed"):
        return _finalize(state, draft, {})

    sentences = v.split_sentences(draft)
    v.check_citations(
        sentences, len(state["reranked_chunks"]),
        restrict_dose=state.get("safety_label") == "medication_dosing",
    )
    await _verify(state, sentences)
    counts = v.claim_counts(sentences)
    kept, removed = v.split_kept_removed(sentences)

    if v.needs_regeneration(sentences) and not state.get("regenerations"):
        return {
            "regenerate": True, "regenerations": 1, "claims": counts,
            "rejected_sentences": [s.text for s in removed], "llm_calls": state["llm_calls"],
        }

    if not kept:
        return _finalize(state, NO_INFO, {"claims": counts, "sources": [], "evidence_status": "insufficient"})
    text = v.rebuild(kept)
    if removed and state.get("regenerations"):
        text += FILTERED_NOTE
    return _finalize(state, text, {"claims": counts})


def finalize_answer(state: State) -> dict:
    """Thay citation_validator khi tắt validator (ablation): chốt bản nháp nguyên trạng."""
    return {
        "final_answer": state["draft_answer"], "regenerate": False,
        "messages": [AIMessage(content=state["draft_answer"])], "llm_calls": state["llm_calls"],
    }
