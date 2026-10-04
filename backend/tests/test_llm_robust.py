"""Parse JSON của LLM chịu được các lỗi dạng hay gặp: code fence, object thay vì danh sách."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from medical_agentic_rag.errors import LLMTaskFailed
from medical_agentic_rag.graph.nodes import analyze as analyze_mod
from medical_agentic_rag.graph.nodes import validate as validate_mod
from medical_agentic_rag.llm.client import strip_code_fence
from medical_agentic_rag.llm.schemas import ChunkJudgement, QueryAnalysis, Rewrites
from medical_agentic_rag.safety.templates import EMERGENCY_TOPIC_NOTE


def test_boc_code_fence():
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence('```\n{"a": 1}\n```  ') == '{"a": 1}'
    assert strip_code_fence('{"a": 1}') == '{"a": 1}'
    assert strip_code_fence(None) == ""


def test_analysis_nhan_object_thay_cho_danh_sach():
    a = QueryAnalysis.model_validate_json(
        '{"question_type": "definition", "safety_label": "normal", "entities": "zona", "aspects": ["x"],'
        ' "target_section_types": {"xử trí cấp cứu": "treatment", "triệu chứng": "symptom"}}')
    assert a.target_section_types == ["treatment", "symptom"] and a.entities == ["zona"]


def test_analysis_van_nhan_danh_sach_binh_thuong():
    a = QueryAnalysis.model_validate_json(
        '{"question_type": "definition", "safety_label": "normal", "entities": ["a"], "aspects": ["b"],'
        ' "target_section_types": ["treatment"]}')
    assert a.target_section_types == ["treatment"]


def test_grade_va_rewrite_cung_ep_kieu():
    assert ChunkJudgement(n=1, relevant=True, covers="nguyên nhân").covers == ["nguyên nhân"]
    assert Rewrites.model_validate({"queries": {"a": "câu một"}, "rationale": "r"}).queries == ["câu một"]


def test_analyze_thoai_hoa_thi_danh_dau_safety_chua_kiem_tra(monkeypatch):
    async def boom(*a, **k):
        raise LLMTaskFailed("down")

    monkeypatch.setattr(analyze_mod, "run_task", boom)
    out = asyncio.run(analyze_mod.analyze_query({"standalone_question": "q", "llm_calls": 0}))
    assert out["safety_unverified"] is True and out["question_type"] == ""


def test_safety_chua_kiem_tra_thi_chen_khuyen_cao_115():
    out = validate_mod._finalize({"llm_calls": 0, "safety_label": "normal", "safety_unverified": True}, "Nội dung [1].", {})
    assert out["final_answer"].startswith(EMERGENCY_TOPIC_NOTE)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
