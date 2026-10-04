"""Test citation validator (mục 10.2): bước 1 + chính sách (code thuần) và node validate_citations."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from medical_agentic_rag.citation import validator as v
from medical_agentic_rag.errors import LLMTaskFailed
from medical_agentic_rag.graph.nodes import validate as validate_mod
from medical_agentic_rag.llm.schemas import ClaimVerdict, ClaimVerdicts
from medical_agentic_rag.safety.templates import HIGH_RISK_PREFIX

CHUNKS = [{"chunk_id": "a#00", "text": "Zona do virus varicella-zoster tái hoạt động."},
          {"chunk_id": "b#00", "text": "Tiêm vắc xin giúp phòng ngừa zona."}]


def prepared(text, n=2, **kw):
    s = v.split_sentences(text)
    v.check_citations(s, n, **kw)
    return s


def test_tach_cau_giu_cau_truc_dong():
    text = "Zona do virus gây ra [1]. Bệnh gây đau rát [1].\n- Nên tiêm vắc xin phòng bệnh [2]."
    s = v.split_sentences(text)
    assert [x.line for x in s] == [0, 0, 1]
    assert v.rebuild(s) == text


def test_xoa_ref_la_va_danh_dau_cau_thieu_nguon():
    s = prepared("Zona do virus varicella gây ra [1][9]. Bệnh này gây đau rát dữ dội ở vùng da.")
    assert s[0].refs == [1] and "[9]" not in s[0].text and not s[0].verdict
    assert s[1].verdict == "unsupported" and s[1].reason == "không có trích dẫn"


def test_cau_chi_co_ref_la_thanh_thieu_nguon():
    s = prepared("Zona do virus varicella gây ra bệnh [7].")
    assert s[0].verdict == "unsupported"


def test_cau_khuyen_cao_khong_can_nguon():
    s = prepared("YouMed chưa có thông tin về phần này. Bạn nên đi khám bác sĩ để được tư vấn.")
    assert not any(x.needs_cite for x in s) and not any(x.verdict for x in s)


def test_loai_cau_chua_lieu_thuoc_khi_hoi_lieu():
    s = prepared("Paracetamol giúp hạ sốt [1]. Bạn nên uống 500 mg mỗi 6 giờ [1].", restrict_dose=True)
    assert s[0].verdict == "" and s[1].verdict == "unsupported" and "liều" in s[1].reason


def test_chinh_sach_xoa_tren_30_phan_tram_thi_viet_lai():
    s = prepared("Ý một có nguồn rõ ràng [1]. Ý hai có nguồn rõ ràng [1]. Ý ba có nguồn rõ ràng [2].")
    assert not v.needs_regeneration(s)
    s[0].verdict = "unsupported"  # 1/3 = 33% > 30%
    assert v.needs_regeneration(s)


def test_chinh_sach_xoa_it_thi_chi_loc():
    s = prepared(" ".join(f"Ý số {i} có nguồn rõ ràng [1]." for i in range(5)))
    s[0].verdict = "unsupported"  # 1/5 = 20%
    assert not v.needs_regeneration(s)
    kept, removed = v.split_kept_removed(s)
    assert len(kept) == 4 and len(removed) == 1


def _node_state(draft, **over):
    state = {
        "draft_answer": draft, "reranked_chunks": CHUNKS, "llm_calls": 0, "regenerations": 0,
        "safety_label": "normal",
    }
    state.update(over)
    return state


def _fake_verify(verdicts):
    async def run_task(task, prompt, state, schema=None, tools=None):
        assert task == "verify_claims"
        return ClaimVerdicts(verdicts=[ClaimVerdict(sentence_id=i, verdict=vd, reason="") for i, vd in verdicts.items()])
    return run_task


def test_node_tat_ca_supported_chot_ban_nhap(monkeypatch):
    monkeypatch.setattr(validate_mod, "run_task", _fake_verify({0: "supported", 1: "partial"}))
    draft = "Zona do virus varicella-zoster tái hoạt động [1]. Tiêm vắc xin giúp phòng ngừa zona [2]."
    out = asyncio.run(validate_mod.validate_citations(_node_state(draft)))
    assert out["final_answer"] == draft and out["regenerate"] is False
    assert out["claims"] == {"supported": 1, "partial": 1, "unsupported": 0}


def test_node_xoa_nhieu_thi_yeu_cau_viet_lai_roi_loc_kem_ghi_chu(monkeypatch):
    monkeypatch.setattr(validate_mod, "run_task", _fake_verify({0: "unsupported", 1: "supported"}))
    draft = "Zona gây ra bởi nấm men trong ruột [1]. Tiêm vắc xin giúp phòng ngừa zona [2]."
    first = asyncio.run(validate_mod.validate_citations(_node_state(draft)))
    assert first["regenerate"] is True and first["regenerations"] == 1
    assert first["rejected_sentences"] == ["Zona gây ra bởi nấm men trong ruột [1]."]

    second = asyncio.run(validate_mod.validate_citations(_node_state(draft, regenerations=1)))
    assert second["regenerate"] is False
    assert second["final_answer"].startswith("Tiêm vắc xin") and "chưa đủ nguồn" in second["final_answer"]


def test_node_loc_het_cau_thi_tra_no_info(monkeypatch):
    monkeypatch.setattr(validate_mod, "run_task", _fake_verify({0: "unsupported"}))
    out = asyncio.run(validate_mod.validate_citations(
        _node_state("Zona gây ra bởi nấm men trong ruột [1].", regenerations=1)))
    assert out["final_answer"] == validate_mod.NO_INFO and out["sources"] == []


def test_node_verify_loi_van_ap_buoc_1(monkeypatch):
    async def boom(*a, **k):
        raise LLMTaskFailed("down")

    monkeypatch.setattr(validate_mod, "run_task", boom)
    draft = "Zona do virus varicella-zoster tái hoạt động [1]. Câu này không có nguồn nhưng nói chuyện y khoa dài."
    out = asyncio.run(validate_mod.validate_citations(_node_state(draft, regenerations=1)))
    assert out["final_answer"].startswith("Zona do virus") and "không có nguồn nhưng" not in out["final_answer"]


def test_node_high_risk_them_khuyen_cao_bang_code(monkeypatch):
    monkeypatch.setattr(validate_mod, "run_task", _fake_verify({0: "supported"}))
    out = asyncio.run(validate_mod.validate_citations(
        _node_state("Zona do virus varicella-zoster tái hoạt động [1].", safety_label="high_risk")))
    assert out["final_answer"].startswith(HIGH_RISK_PREFIX)


def test_node_generate_failed_khong_kiem_tra(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("không được verify khi generate đã lỗi")

    monkeypatch.setattr(validate_mod, "run_task", boom)
    out = asyncio.run(validate_mod.validate_citations(_node_state("Hệ thống tạm thời lỗi.", generate_failed=True)))
    assert out["final_answer"] == "Hệ thống tạm thời lỗi."


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
