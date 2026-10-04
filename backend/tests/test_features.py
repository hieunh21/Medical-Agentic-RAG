"""Cấu hình ablation (Features): topology đúng, và chạy được end-to-end bằng mock (không network)."""
import asyncio
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from medical_agentic_rag.graph import build as build_mod
from medical_agentic_rag.graph.build import Features, compile_graph
from medical_agentic_rag.graph.nodes import generate as generate_mod
from medical_agentic_rag.graph.nodes import retrieve as retrieve_mod
from medical_agentic_rag.graph.state import init_state

CHUNK = {
    "chunk_id": "a#00", "article_id": "a", "title": "Zona", "url": "u", "section_type": "overview",
    "text": "Zona do virus varicella-zoster tái hoạt động.",
}


def reachable(graph) -> set[str]:
    g = graph.get_graph()
    seen, stack = set(), ["__start__"]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(e.target for e in g.edges if e.source == n)
    return seen


def test_moi_to_hop_hop_le_deu_compile_duoc():
    n = 0
    for r, c, a, s, v in itertools.product([True, False], repeat=5):
        if s and not r:
            continue
        compile_graph(features=Features(routing=r, corrective=c, agent=a, safety=s, validator=v))
        n += 1
    assert n == 24


def test_safety_can_routing():
    with pytest.raises(ValueError):
        Features(routing=False, safety=True)


def test_baseline_p1_khong_co_thanh_phan_nao_khac():
    nodes = reachable(compile_graph(features=Features(
        routing=False, corrective=False, agent=False, safety=False, validator=False)))
    assert nodes == {"__start__", "hybrid_retrieve", "rerank", "generate_answer", "citation_validator", "__end__"}


def test_day_du_co_moi_thanh_phan():
    nodes = reachable(compile_graph())
    for n in ("safety_rules", "analyze_query", "grade_evidence", "research_agent", "citation_validator", "safety_response"):
        assert n in nodes


def test_tat_agent_thi_khong_toi_duoc_research_agent():
    assert "research_agent" not in reachable(compile_graph(features=Features(agent=False)))


def test_tat_safety_thi_khong_toi_duoc_safety_response():
    assert "safety_response" not in reachable(compile_graph(features=Features(safety=False)))


def test_baseline_chay_end_to_end_bang_mock(monkeypatch):
    monkeypatch.setattr(retrieve_mod, "retrieve", lambda q: [dict(CHUNK, score=1.0)])
    monkeypatch.setattr(retrieve_mod, "rerank_and_select", lambda q, cs: [dict(c, rerank_score=1.0) for c in cs])

    async def fake_run_task(task, prompt, state, schema=None, tools=None):
        assert task == "generate_answer"  # baseline chỉ được gọi LLM đúng 1 lần, để generate
        assert "chỉ trả lời được một phần" not in prompt  # chưa qua grader -> không gắn nhãn partial
        state["llm_calls"] = state.get("llm_calls", 0) + 1
        return "Zona do virus varicella-zoster tái hoạt động [1]."

    monkeypatch.setattr(generate_mod, "run_task", fake_run_task)
    graph = compile_graph(features=Features(
        routing=False, corrective=False, agent=False, safety=False, validator=False))
    result = asyncio.run(graph.ainvoke(init_state("Zona là gì?")))
    assert result["final_answer"].endswith("[1].") and result["llm_calls"] == 1
    assert result["sources"][0]["title"] == "Zona"


def test_tat_generate_thi_dung_sau_retrieval():
    full = reachable(compile_graph(features=Features(generate=False)))
    assert "generate_answer" not in full and "citation_validator" not in full
    assert "grade_evidence" in full and "research_agent" in full  # phần retrieval vẫn đủ


def test_baseline_retrieval_only_khong_goi_llm(monkeypatch):
    monkeypatch.setattr(retrieve_mod, "retrieve", lambda q: [dict(CHUNK, score=1.0)])
    monkeypatch.setattr(retrieve_mod, "rerank_and_select", lambda q, cs: [dict(c, rerank_score=1.0) for c in cs])

    async def boom(*a, **k):
        raise AssertionError("retrieval-only không được gọi LLM")

    monkeypatch.setattr(generate_mod, "run_task", boom)
    graph = compile_graph(features=Features(
        routing=False, corrective=False, agent=False, safety=False, validator=False, generate=False))
    result = asyncio.run(graph.ainvoke(init_state("Zona là gì?")))
    assert result["reranked_chunks"][0]["chunk_id"] == "a#00" and not result.get("final_answer")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
