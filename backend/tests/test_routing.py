"""Test luật dừng after_grade (mục 7.4) — chạy bằng pytest, không cần network/model server."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from medical_agentic_rag.config import settings
from medical_agentic_rag.graph.routing import after_grade


def base_state(**over) -> dict:
    state = {"llm_calls": 0, "corrections": 0, "n_relevant": 0, "coverage": 0.0}
    state.update(over)
    return state


def test_coverage_du_thi_tra_loi_ngay():
    s = base_state(n_relevant=3, coverage=settings.COVERAGE_THRESHOLD)
    assert after_grade(s) == "generate_answer"


def test_khong_co_chunk_lien_quan_thi_rewrite():
    s = base_state(n_relevant=0, corrections=0)
    assert after_grade(s) == "rewrite_query"


def test_het_luot_rewrite_ma_van_khong_co_gi_thi_no_info():
    s = base_state(n_relevant=0, corrections=settings.MAX_CORRECTIONS)
    assert after_grade(s) == "no_info_response"


def test_co_lien_quan_nhung_thieu_khia_canh_thi_targeted_retrieve():
    s = base_state(n_relevant=2, coverage=0.0, corrections=0)
    assert after_grade(s) == "targeted_retrieve"


def test_het_luot_sua_ma_coverage_van_thap_thi_tra_loi_partial():
    s = base_state(n_relevant=2, coverage=0.0, corrections=settings.MAX_CORRECTIONS)
    assert after_grade(s) == "generate_answer"


def test_het_ngan_sach_llm_va_co_chunk_lien_quan_thi_tra_loi_ngay():
    s = base_state(n_relevant=1, coverage=0.0, llm_calls=settings.MAX_LLM_CALLS_PER_REQUEST)
    assert after_grade(s) == "generate_answer"


def test_het_ngan_sach_llm_va_khong_co_chunk_nao_thi_no_info():
    s = base_state(n_relevant=0, llm_calls=settings.MAX_LLM_CALLS_PER_REQUEST)
    assert after_grade(s) == "no_info_response"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
