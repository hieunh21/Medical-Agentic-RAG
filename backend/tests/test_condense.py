"""condense_question: lịch sử đưa cho LLM phải theo settings.HISTORY_TURNS, không hard-code."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from medical_agentic_rag.config import settings
from medical_agentic_rag.graph.nodes import condense as condense_mod
from medical_agentic_rag.llm.schemas import Condensed


def convo(n_turns: int) -> list:
    """n lượt đã xong + 1 câu hỏi mới ở cuối (chính câu đang xử lý)."""
    msgs = []
    for i in range(n_turns):
        msgs.append(HumanMessage(content=f"câu hỏi {i}"))
        msgs.append(AIMessage(content=f"trả lời {i}"))
    msgs.append(HumanMessage(content="còn triệu chứng thì sao?"))
    return msgs


def run(messages, monkeypatch) -> str:
    seen = {}

    async def fake(task, prompt, state, schema=None, tools=None, on_token=None):
        seen["prompt"] = prompt
        return Condensed(standalone_question="q độc lập", is_followup=True, carried_entities=[])

    monkeypatch.setattr(condense_mod, "run_task", fake)
    monkeypatch.setattr(condense_mod, "get_titles", lambda ids: [])
    out = asyncio.run(condense_mod.condense_question(
        {"messages": messages, "question": "còn triệu chứng thì sao?", "llm_calls": 0}))
    assert out["standalone_question"] == "q độc lập"
    return seen["prompt"]


def test_luot_dau_khong_goi_llm(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("lượt đầu không được gọi LLM")

    monkeypatch.setattr(condense_mod, "run_task", boom)
    out = asyncio.run(condense_mod.condense_question(
        {"messages": [HumanMessage(content="Zona là gì?")], "question": "Zona là gì?", "llm_calls": 0}))
    assert out == {"standalone_question": "Zona là gì?", "is_followup": False}


def test_cat_dung_so_luot_theo_settings(monkeypatch):
    monkeypatch.setattr(settings, "HISTORY_TURNS", 2)
    prompt = run(convo(5), monkeypatch)
    assert "câu hỏi 3" in prompt and "câu hỏi 4" in prompt  # 2 lượt gần nhất
    assert "câu hỏi 2" not in prompt                        # lượt cũ hơn bị cắt


def test_doi_settings_thi_so_luot_doi_theo(monkeypatch):
    monkeypatch.setattr(settings, "HISTORY_TURNS", 4)
    prompt = run(convo(5), monkeypatch)
    assert "câu hỏi 1" in prompt   # trước đây bị chặn ở 3 lượt nên dòng này không có
    assert "câu hỏi 0" not in prompt


def test_cat_cau_tra_loi_dai(monkeypatch):
    long_answer = "x" * 900
    prompt = run([HumanMessage(content="hỏi"), AIMessage(content=long_answer), HumanMessage(content="rồi sao")],
                 monkeypatch)
    assert "x" * condense_mod_preview() in prompt and "x" * (condense_mod_preview() + 1) not in prompt


def condense_mod_preview() -> int:
    from medical_agentic_rag.llm.prompts import ANSWER_PREVIEW_CHARS
    return ANSWER_PREVIEW_CHARS


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
