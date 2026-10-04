"""section_lookup: gộp ứng viên lọc theo bài với ứng viên không lọc, để bài "anh em" không bị mất."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from medical_agentic_rag.graph.nodes import route_retrieve as rr


def chunk(cid, article):
    return {"chunk_id": cid, "article_id": article, "url": f"u/{article}", "text": cid}


@pytest.fixture
def calls(monkeypatch):
    log = []

    def fake_retrieve(q, limit=None, section_type=None, article_id=None):
        log.append({"article_id": article_id, "section_type": section_type})
        if article_id == "a":
            return [chunk("a#01", "a")] if section_type else [chunk("a#01", "a"), chunk("a#02", "a")]
        return [chunk("b#01", "b"), chunk("a#01", "a")]  # không lọc: có cả bài "anh em" b

    monkeypatch.setattr(rr, "retrieve", fake_retrieve)
    monkeypatch.setattr(rr, "find_article", lambda name: {"article_id": "a"} if name != "không rõ" else None)
    monkeypatch.setattr(rr, "rerank_and_select", lambda q, cs: cs)
    return log


STATE = {"standalone_question": "q", "entities": ["bệnh a"], "target_section_types": ["symptom"]}


def test_gop_ung_vien_loc_va_khong_loc(calls):
    pool, chunks = rr._section_lookup_sync(STATE)
    assert set(pool) == {"a#01", "b#01"}  # bài b chỉ có ở lượt không lọc
    assert calls == [{"article_id": "a", "section_type": "symptom"}, {"article_id": None, "section_type": None}]


def test_widen_false_la_hanh_vi_cu(calls):
    pool, _ = rr._section_lookup_sync(STATE, widen=False)
    assert set(pool) == {"a#01"} and len(calls) == 1


def test_khong_khop_bai_thi_chi_mot_lan_khong_loc(calls):
    pool, _ = rr._section_lookup_sync({**STATE, "entities": ["không rõ"]})
    assert set(pool) == {"b#01", "a#01"} and calls == [{"article_id": None, "section_type": None}]


def test_loc_section_rong_thi_thu_lai_chi_loc_bai(monkeypatch, calls):
    def fake_retrieve(q, limit=None, section_type=None, article_id=None):
        calls.append({"article_id": article_id, "section_type": section_type})
        if article_id == "a" and section_type:
            return []
        return [chunk("a#02", "a")] if article_id == "a" else [chunk("b#01", "b")]

    monkeypatch.setattr(rr, "retrieve", fake_retrieve)
    calls.clear()
    pool, _ = rr._section_lookup_sync(STATE)
    assert set(pool) == {"a#02", "b#01"}
    assert calls[1] == {"article_id": "a", "section_type": None}  # lượt 2: bỏ section_type, giữ bài


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
