"""Test ngân sách LLM: generate_answer luôn còn chỗ, node bọc degrade_on_budget không làm sập request."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from medical_agentic_rag import budget
from medical_agentic_rag.config import settings
from medical_agentic_rag.errors import LLMTaskFailed
from medical_agentic_rag.graph.nodes import grade as grade_mod
from medical_agentic_rag.graph.nodes import generate as generate_mod
from medical_agentic_rag.graph.nodes.grade import grade_evidence
from medical_agentic_rag.graph.routing import after_grade

MAX = settings.MAX_LLM_CALLS_PER_REQUEST


def test_task_thuong_bi_chan_o_max_tru_1():
    with pytest.raises(budget.BudgetExceeded):
        budget.charge_llm({"llm_calls": MAX - 1}, "grade_evidence")


def test_generate_answer_van_con_luot_cuoi():
    state = {"llm_calls": MAX - 1}
    budget.charge_llm(state, "generate_answer")
    assert state["llm_calls"] == MAX


def test_exhausted_khi_chi_con_luot_cho_generate():
    assert not budget.exhausted({"llm_calls": MAX - 2})
    assert budget.exhausted({"llm_calls": MAX - 1})


def test_grade_het_ngan_sach_thi_thoai_hoa_thay_vi_nem_loi():
    state = {
        "llm_calls": MAX - 1, "aspects": ["a"], "standalone_question": "q",
        "reranked_chunks": [{"chunk_id": "x#00", "article_id": "x", "title": "t", "url": "u", "text": "t"}],
    }
    out = asyncio.run(grade_evidence(state))
    assert out["n_relevant"] == 1 and out["coverage"] == 0.0 and out["missing_aspects"] == ["a"]


def _boom(*a, **k):
    raise LLMTaskFailed("api down")


def test_grade_llm_loi_thi_thoai_hoa_va_di_thang_toi_generate(monkeypatch):
    monkeypatch.setattr(grade_mod, "run_task", _boom)
    state = {
        "llm_calls": 0, "aspects": ["a"], "standalone_question": "q", "corrections": 0,
        "reranked_chunks": [{"chunk_id": "x#00", "article_id": "x", "title": "t", "url": "u", "text": "t"}],
    }
    out = asyncio.run(grade_evidence(state))
    assert out["grade_failed"] is True
    assert after_grade({**state, **out}) == "generate_answer"  # không lặp vòng sửa


def test_generate_llm_loi_thi_tra_thong_bao_kem_nguon(monkeypatch):
    monkeypatch.setattr(generate_mod, "run_task", _boom)
    chunk = {"chunk_id": "x#00", "article_id": "x", "title": "t", "url": "u", "text": "t"}
    out = asyncio.run(generate_mod.generate_answer_node(
        {"reranked_chunks": [chunk], "standalone_question": "q", "llm_calls": 0}, {}))
    assert out["draft_answer"] == generate_mod.GENERATE_FAILED and out["generate_failed"] is True
    assert out["evidence_status"] == "insufficient" and out["sources"]



if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
