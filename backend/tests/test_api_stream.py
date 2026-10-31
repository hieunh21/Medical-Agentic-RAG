"""SSE của /api/v1/chat/stream: nhãn trạng thái phát lúc node BẮT ĐẦU, nguồn sớm đánh số theo bài."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
from langgraph.graph import END, StateGraph

from medical_agentic_rag.api import app as app_mod
from medical_agentic_rag.graph.state import State


def parse(raw: str) -> tuple[str, dict]:
    lines = raw.strip().split("\n")
    event = lines[0].removeprefix("event: ")
    return event, json.loads(lines[1].removeprefix("data: "))


@pytest.fixture(autouse=True)
def no_trace(monkeypatch):
    monkeypatch.setattr(app_mod.trace, "log_request", lambda *a, **k: None)


def use_graph(nodes: list[tuple[str, object]]):
    g = StateGraph(State)
    for name, fn in nodes:
        g.add_node(name, fn)
    g.set_entry_point(nodes[0][0])
    for (a, _), (b, _) in zip(nodes, nodes[1:]):
        g.add_edge(a, b)
    g.add_edge(nodes[-1][0], END)
    app_mod.app.state.graph = g.compile()


def test_nhan_trang_thai_phat_luc_node_bat_dau():
    """Node chỉ chạy xong khi test đã thấy nhãn của chính nó. Nếu nhãn phát lúc node XONG
    (hành vi cũ) thì node chờ mãi -> hết giờ -> test fail."""
    gate = asyncio.Event()
    label = app_mod.STEP_LABELS["analyze_query"]

    async def slow(state):
        await asyncio.wait_for(gate.wait(), timeout=5)
        return {"question_type": "definition"}

    use_graph([("analyze_query", slow)])

    async def run():
        seen = []
        async for raw in app_mod._chat_events("q", "s1"):
            event, data = parse(raw)
            seen.append((event, data))
            if event == "status" and data["label"] == label:
                gate.set()
        return seen

    seen = asyncio.run(asyncio.wait_for(run(), timeout=10))
    assert ("status", {"step": "analyze_query", "label": label}) in seen
    assert seen[-1][0] == "done"  # node đã chạy xong bình thường, không phải lỗi hết giờ


def test_nguon_som_danh_so_theo_bai_khong_theo_doan():
    chunks = [
        {"chunk_id": "a#00", "article_id": "a", "title": "Zona", "url": "u/a", "text": "x"},
        {"chunk_id": "a#05", "article_id": "a", "title": "Zona", "url": "u/a", "text": "y"},
        {"chunk_id": "b#00", "article_id": "b", "title": "Vắc xin", "url": "u/b", "text": "z"},
    ]

    async def retrieve(state):
        return {"reranked_chunks": chunks}

    use_graph([("hybrid_retrieve", retrieve)])

    async def run():
        return [parse(raw) async for raw in app_mod._chat_events("q", "s2")]

    sources = next(d["sources"] for e, d in asyncio.run(run()) if e == "sources")
    assert [s["n"] for s in sources] == [1, 2] and [s["url"] for s in sources] == ["u/a", "u/b"]


def test_ket_qua_cuoi_giu_trace_id_cua_input(monkeypatch):
    """"updates" chỉ chứa phần node trả về; trace_id nằm ở input nên trước đây bị mất khỏi trace."""
    logged = {}
    monkeypatch.setattr(app_mod.trace, "log_request", lambda result, ms: logged.update(result))

    async def node(state):
        return {"question_type": "definition"}

    use_graph([("analyze_query", node)])

    async def run():
        return [raw async for raw in app_mod._chat_events("q", "s3")]

    asyncio.run(run())
    assert logged.get("trace_id") and logged.get("thread_id") == "s3"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
