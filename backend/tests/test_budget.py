"""Test ngân sách LLM: generate_answer luôn còn chỗ, node bọc degrade_on_budget không làm sập request."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from medical_agentic_rag import budget
from medical_agentic_rag.config import settings
from medical_agentic_rag.graph.nodes.grade import grade_evidence

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
        "reranked_chunks": [{"chunk_id": "x#00", "title": "t", "text": "t"}],
    }
    out = asyncio.run(grade_evidence(state))
    assert out["n_relevant"] == 1 and out["coverage"] == 0.0 and out["missing_aspects"] == ["a"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
