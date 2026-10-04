"""sub_results (reducer cộng dồn) không được rò từ lượt này sang lượt sau trong cùng thread."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from medical_agentic_rag.graph.state import State, accumulate_results, init_state


def test_reducer_noi_va_reset():
    assert accumulate_results(None, [{"a": 1}]) == [{"a": 1}]
    assert accumulate_results([{"a": 1}], [{"b": 2}]) == [{"a": 1}, {"b": 2}]
    assert accumulate_results([{"a": 1}], None) == []
    assert accumulate_results([{"a": 1}], []) == [{"a": 1}]


def test_hai_luot_cung_thread_khong_cong_don_sub_results():
    seen = []

    def branch_a(state):
        return {"sub_results": [{"q": "a"}]}

    def branch_b(state):
        return {"sub_results": [{"q": "b"}]}

    def merge(state):
        seen.append(len(state["sub_results"]))
        return {}

    g = StateGraph(State)
    g.add_node("a", branch_a)
    g.add_node("b", branch_b)
    g.add_node("merge", merge)
    g.set_entry_point("a")
    g.add_edge("a", "b")
    g.add_edge("b", "merge")
    g.add_edge("merge", END)
    graph = g.compile(checkpointer=MemorySaver())

    cfg = {"configurable": {"thread_id": "t1"}}
    asyncio.run(graph.ainvoke(init_state("lượt 1", thread_id="t1"), config=cfg))
    asyncio.run(graph.ainvoke(init_state("lượt 2", thread_id="t1"), config=cfg))
    assert seen == [2, 2]  # lượt 2 không thấy 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
