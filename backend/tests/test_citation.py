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

CHUNKS = [
    {"chunk_id": "a#00", "article_id": "a", "title": "Zona", "url": "u/a",
     "text": "Zona do virus varicella-zoster tái hoạt động."},
    {"chunk_id": "b#00", "article_id": "b", "title": "Vắc xin zona", "url": "u/b",
     "text": "Tiêm vắc xin giúp phòng ngừa zona."},
]


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


# --------------------------------------------------------------------------- nhiều nguồn trong 1 cặp ngoặc
def test_nhan_dang_tri_ch_dan_gop_nhieu_nguon():
    s = prepared("Zona do virus varicella gây ra [1, 2]. Tiêm vắc xin giúp phòng ngừa [1,2].")
    assert s[0].refs == [1, 2] and s[1].refs == [1, 2]
    assert not s[0].verdict and not s[1].verdict  # KHÔNG bị coi là thiếu trích dẫn


def test_gop_nhieu_nguon_loc_so_khong_co_that():
    s = prepared("Zona do virus varicella-zoster gây ra bệnh [1, 9].")
    assert s[0].refs == [1] and "[1]" in s[0].text and "9" not in s[0].text and not s[0].verdict


def test_gop_nhieu_nguon_ma_khong_so_nao_hop_le_thi_unsupported():
    s = prepared("Zona do virus varicella-zoster gây ra bệnh [8, 9].")
    assert s[0].refs == [] and s[0].verdict == "unsupported"


def test_khong_pha_cau_truc_khi_viet_lai_nhom():
    s = prepared("Zona do virus varicella gây ra [2, 1].")
    assert s[0].text == "Zona do virus varicella gây ra [2][1]."  # giữ thứ tự model viết


# --------------------------------------------------------------------------- nguồn trùng / không trích
DUP = [
    {"chunk_id": "a#00", "article_id": "a", "title": "Zona", "url": "u/a", "text": "Đoạn 1 của bài A."},
    {"chunk_id": "a#05", "article_id": "a", "title": "Zona", "url": "u/a", "text": "Đoạn 2 của bài A."},
    {"chunk_id": "b#00", "article_id": "b", "title": "Vắc xin", "url": "u/b", "text": "Đoạn của bài B."},
]


def test_hai_doan_cung_bai_chi_thanh_mot_nguon():
    from medical_agentic_rag.answer import format_context, format_sources

    assert [s["n"] for s in format_sources(DUP)] == [1, 2]
    assert [s["url"] for s in format_sources(DUP)] == ["u/a", "u/b"]  # không còn u/a hai lần
    ctx = format_context(DUP)
    assert "[1]" in ctx and "[2]" in ctx and "[3]" not in ctx
    assert "Đoạn 1 của bài A." in ctx and "Đoạn 2 của bài A." in ctx  # cả 2 đoạn vẫn vào context


def test_bo_nguon_khong_duoc_trich_va_danh_so_lai():
    sources = [{"n": i, "title": f"Bài {i}", "url": f"u/{i}", "updated_date": None} for i in range(1, 6)]
    s = prepared("Ý một lấy từ nguồn một [1]. Ý hai lấy từ nguồn năm [5].", n=5)
    kept = v.renumber_citations(s, sources)
    assert [x["n"] for x in kept] == [1, 2]
    assert [x["url"] for x in kept] == ["u/1", "u/5"]        # giữ đúng 2 bài được trích
    assert s[0].refs == [1] and s[1].refs == [2]             # [5] -> [2]
    assert v.rebuild(s) == "Ý một lấy từ nguồn một [1]. Ý hai lấy từ nguồn năm [2]."


def test_danh_so_lai_cho_tri_ch_dan_gop_nhieu_nguon():
    sources = [{"n": i, "title": f"Bài {i}", "url": f"u/{i}", "updated_date": None} for i in range(1, 6)]
    s = prepared("Ý này lấy từ hai nguồn khác nhau [2, 4].", n=5)
    kept = v.renumber_citations(s, sources)
    assert [x["url"] for x in kept] == ["u/2", "u/4"]
    assert s[0].text == "Ý này lấy từ hai nguồn khác nhau [1][2]."


def test_node_tra_ve_nguon_da_loc(monkeypatch):
    monkeypatch.setattr(validate_mod, "run_task", _fake_verify({0: "supported"}))
    state = {
        "draft_answer": "Đoạn 1 của bài A nói điều này [1].", "reranked_chunks": DUP,
        "llm_calls": 0, "regenerations": 0, "safety_label": "normal",
    }
    out = asyncio.run(validate_mod.validate_citations(state))
    assert [s["url"] for s in out["sources"]] == ["u/a"]  # bài B không được trích -> không hiện
    assert out["final_answer"] == "Đoạn 1 của bài A nói điều này [1]."
