"""Test routing theo question_type (mục 8.3) — 7 route, chạy bằng pytest, không cần network."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from medical_agentic_rag.config import settings
from medical_agentic_rag.graph.routing import after_definition_retrieve, route_by_type


def test_definition_route():
    assert route_by_type({"question_type": "definition"}) == "retrieve_definition"


def test_section_lookup_route():
    assert route_by_type({"question_type": "section_lookup"}) == "retrieve_section_lookup"


def test_symptom_to_condition_route():
    assert route_by_type({"question_type": "symptom_to_condition"}) == "hybrid_retrieve"


def test_comparison_route():
    # Đổi về hybrid_retrieve sau khi eval/run_routing_compare.py cho thấy subquery
    # không vượt baseline đơn giản cho comparison (xem routing.py).
    assert route_by_type({"question_type": "comparison"}) == "hybrid_retrieve"


def test_multi_aspect_route():
    assert route_by_type({"question_type": "multi_aspect"}) == "plan_subqueries"


def test_complex_route():
    assert route_by_type({"question_type": "complex"}) == "research_agent"


def test_complex_route_fallback_khi_agent_bi_tat():
    # RESEARCH_AGENT_ENABLED=false -> rơi về hybrid_retrieve (xem config.py, routing.py)
    old = settings.AGENT_ENABLED
    settings.AGENT_ENABLED = False
    try:
        assert route_by_type({"question_type": "complex"}) == "hybrid_retrieve"
    finally:
        settings.AGENT_ENABLED = old


def test_out_of_scope_route():
    assert route_by_type({"question_type": "out_of_scope"}) == "out_of_scope_response"


def test_unknown_type_fallback_ve_hybrid_retrieve():
    assert route_by_type({"question_type": "khong-ro"}) == "hybrid_retrieve"
    assert route_by_type({}) == "hybrid_retrieve"


def test_definition_fast_path_bo_qua_grader_khi_tu_tin():
    assert after_definition_retrieve({"confident": True}) == "generate_answer"


def test_definition_khong_tu_tin_thi_qua_grader():
    assert after_definition_retrieve({"confident": False}) == "grade_evidence"
    assert after_definition_retrieve({}) == "grade_evidence"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
