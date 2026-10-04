"""Test safety (mục 10.1): rule lớp 1, hợp nhất nhãn, routing dừng pipeline, template."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from medical_agentic_rag.errors import LLMTaskFailed
from medical_agentic_rag.graph import build as build_mod
from medical_agentic_rag.graph.nodes import analyze as analyze_mod
from medical_agentic_rag.graph.nodes import validate as validate_mod
from medical_agentic_rag.graph.nodes.safety import safety_rules
from medical_agentic_rag.llm.schemas import QueryAnalysis
from medical_agentic_rag.graph.routing import after_safety_rules, route_by_type
from medical_agentic_rag.graph.state import init_state
from medical_agentic_rag.safety import templates
from medical_agentic_rag.safety.rules import STOP_LABELS, has_personal_marker, more_severe, rule_label


@pytest.mark.parametrize("text", [
    "Tôi bị đau ngực dữ dội từ sáng", "bố tôi khó thở và co giật", "Con tôi bị ngất", "em bị méo miệng",
    "hôm qua uống nhầm thuốc, nghi ngộ độc", "nôn ra máu phải làm sao",
])
def test_emergency_co_dau(text):
    assert rule_label(text) == "emergency"


@pytest.mark.parametrize("text", ["toi bi dau nguc du doi", "con toi kho tho", "bat tinh nhan su", "yeu nua nguoi ben trai"])
def test_emergency_khong_dau(text):
    assert rule_label(text) == "emergency"


@pytest.mark.parametrize("text", ["Tôi muốn chết", "toi muon chet", "tôi đang nghĩ đến chuyện tự tử"])
def test_self_harm(text):
    assert rule_label(text) == "self_harm"


def test_self_harm_uu_tien_hon_emergency():
    assert rule_label("tôi muốn chết, khó thở lắm") == "self_harm"


@pytest.mark.parametrize("text", [
    "Bệnh Addison là gì?", "Triệu chứng của sốt xuất huyết?", "Tôi bị ngạt mũi về đêm",  # ngạt != ngất khi gõ có dấu
    "Cách phòng ngừa zona thần kinh",
])
def test_cau_hoi_binh_thuong(text):
    assert rule_label(text) == "normal"


def test_more_severe_lay_nhan_nghiem_trong_hon():
    assert more_severe("normal", "high_risk") == "high_risk"
    assert more_severe("emergency", "diagnosis_request") == "emergency"
    assert more_severe("self_harm", "emergency") == "self_harm"
    assert more_severe(None, None) == "normal"


def test_routing_dung_pipeline_khi_emergency_hoac_self_harm():
    for label in STOP_LABELS:
        assert after_safety_rules({"safety_label": label}) == "safety_response"
        assert route_by_type({"safety_label": label, "question_type": "definition"}) == "safety_response"
    assert after_safety_rules({"safety_label": "high_risk"}) == "condense_question"
    assert route_by_type({"safety_label": "high_risk", "question_type": "definition"}) == "retrieve_definition"


def test_template_co_so_cap_cuu_va_khong_bia_hotline():
    assert "115" in templates.emergency_response()
    text = templates.self_harm_response()
    assert "115" in text and "Đường dây hỗ trợ" not in text  # resources.yaml chưa có hotline đã xác minh


def test_init_state_reset_safety_label_moi_luot():
    assert init_state("câu hỏi")["safety_label"] == "normal"


def test_graph_emergency_khong_goi_llm_va_khong_retrieve(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("không được gọi LLM / retrieval khi emergency")

    import medical_agentic_rag.llm.client as client
    monkeypatch.setattr(client, "get_client", boom)
    graph = build_mod.compile_graph()
    result = asyncio.run(graph.ainvoke(init_state("Tôi bị đau ngực dữ dội và khó thở")))
    assert result["safety_label"] == "emergency"
    assert "115" in result["final_answer"]
    assert result["sources"] == [] and result.get("llm_calls", 0) == 0


# ----------------------------------------------------------------------------- hoãn emergency kiến thức chung
@pytest.mark.parametrize("text", [
    "Chứng co giật nửa mặt có gây nguy hiểm không?", "Khó thở là triệu chứng của những bệnh nào?",
    "Co giật ở trẻ em là gì, có những loại nào?", "co giat o tre em la gi", "Khó thở ở bà bầu là gì?",
])
def test_khong_co_dau_hieu_ca_nhan(text):
    assert not has_personal_marker(text)


@pytest.mark.parametrize("text", [
    "Con tôi đang co giật, phải làm sao?", "Tôi bị đau ngực dữ dội và khó thở", "Em bé nhà tôi đang co giật",
    "bo toi dao nay hay bi kho tho", "em bi kho tho tu sang nay", "Mẹ tôi nôn ra máu nhiều lần",
])
def test_co_dau_hieu_ca_nhan(text):
    assert has_personal_marker(text)


def test_safety_rules_hoan_emergency_kien_thuc_chung_nhung_khong_hoan_ca_nhan_hay_tu_hai():
    deferred = safety_rules({"question": "Chứng co giật nửa mặt có gây nguy hiểm không?"})
    assert deferred == {"safety_label": "emergency", "safety_deferred": True}
    assert safety_rules({"question": "Con tôi đang co giật"})["safety_deferred"] is False
    assert safety_rules({"question": "Tự tử là gì?"}) == {"safety_label": "self_harm", "safety_deferred": False}


def test_hoan_thi_khong_dung_ngay_nhung_ca_nhan_thi_dung_ngay():
    assert after_safety_rules({"safety_label": "emergency", "safety_deferred": True}) == "condense_question"
    assert after_safety_rules({"safety_label": "emergency", "safety_deferred": False}) == "safety_response"
    assert after_safety_rules({"safety_label": "self_harm", "safety_deferred": False}) == "safety_response"


def analysis(label, situation="personal"):
    return QueryAnalysis(question_type="definition", safety_label=label, situation=situation,
                         entities=[], aspects=[], target_section_types=[])


def run_analyze(monkeypatch, result, prior_label="normal", deferred=False):
    async def fake(task, prompt, state, schema=None, tools=None):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(analyze_mod, "run_task", fake)
    state = {"standalone_question": "q", "llm_calls": 0, "safety_label": prior_label, "safety_deferred": deferred}
    return asyncio.run(analyze_mod.analyze_query(state)), state


def test_llm_noi_kien_thuc_chung_thi_ha_emergency_va_danh_dau_chu_de(monkeypatch):
    out, _ = run_analyze(monkeypatch, analysis("emergency", "general"))
    assert out["safety_label"] == "normal" and out["emergency_topic"] is True
    out, _ = run_analyze(monkeypatch, analysis("normal", "general"), prior_label="emergency", deferred=True)
    assert out["safety_label"] == "normal" and out["emergency_topic"] is True  # rule hoãn + LLM nói chung


def test_llm_noi_tinh_huong_ca_nhan_thi_van_dung(monkeypatch):
    out, _ = run_analyze(monkeypatch, analysis("emergency", "personal"))
    assert out["safety_label"] == "emergency" and out["emergency_topic"] is False
    out, _ = run_analyze(monkeypatch, analysis("normal", "personal"), prior_label="emergency", deferred=True)
    assert out["safety_label"] == "emergency"  # rule hoãn mà LLM không xác nhận là chung -> giữ dừng
    assert route_by_type({**out, "question_type": "definition"}) == "safety_response"


def test_khong_bao_gio_ha_self_harm(monkeypatch):
    out, _ = run_analyze(monkeypatch, analysis("self_harm", "general"))
    assert out["safety_label"] == "self_harm" and out["emergency_topic"] is False


def test_mac_dinh_situation_la_personal():
    assert QueryAnalysis(question_type="definition", safety_label="emergency", entities=[], aspects=[],
                         target_section_types=[]).situation == "personal"


def test_llm_loi_thi_van_dung_voi_emergency_da_hoan(monkeypatch):
    out, state = run_analyze(monkeypatch, LLMTaskFailed("down"), prior_label="emergency", deferred=True)
    merged = {**state, **out}
    assert merged["safety_label"] == "emergency"  # analyze thoái hoá không đụng nhãn của lớp 1
    assert route_by_type(merged) == "safety_response"


def test_chen_khuyen_cao_goi_115_khi_hoi_kien_thuc_chu_de_cap_cuu():
    out = validate_mod._finalize({"llm_calls": 0, "safety_label": "normal", "emergency_topic": True}, "Nội dung [1].", {})
    assert out["final_answer"].startswith(templates.EMERGENCY_TOPIC_NOTE) and "115" in out["final_answer"]
    plain = validate_mod._finalize({"llm_calls": 0, "safety_label": "normal"}, "Nội dung [1].", {})
    assert plain["final_answer"] == "Nội dung [1]."


def test_init_state_reset_co_moi_luot():
    st = init_state("câu hỏi")
    assert st["safety_deferred"] is False and st["emergency_topic"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
