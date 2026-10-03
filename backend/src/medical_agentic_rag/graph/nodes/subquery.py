"""plan_subqueries + retrieve song song bằng Send + merge_evidence (mục 8.4).

Dùng cho comparison / multi_aspect: mỗi subquery chạy hybrid retrieve riêng (song
song), merge_evidence dedupe theo chunk_id, giữ tối đa 3 chunk/subquery rồi rerank
toàn bộ theo câu hỏi gốc.
"""
from __future__ import annotations

from langgraph.types import Send

from medical_agentic_rag.graph.state import State
from medical_agentic_rag.llm.client import run_task
from medical_agentic_rag.llm.prompts import build_plan_subqueries_prompt
from medical_agentic_rag.llm.schemas import SubqueryPlan
from medical_agentic_rag.retrieval.hybrid import retrieve
from medical_agentic_rag.retrieval.rerank import rerank_balanced_by_group

MAX_SUBQUERIES = 4
MAX_CHUNKS_PER_SUBQUERY = 3


async def plan_subqueries(state: State) -> dict:
    prompt = build_plan_subqueries_prompt(
        state["standalone_question"], state.get("entities") or [], state.get("aspects") or [],
    )
    plan: SubqueryPlan = await run_task("plan_subqueries", prompt, state, schema=SubqueryPlan)

    subqueries = plan.subqueries[:MAX_SUBQUERIES] or [state["standalone_question"]]
    section_types = list(plan.section_types[: len(subqueries)])
    while len(section_types) < len(subqueries):
        section_types.append(None)

    return {
        "subqueries": subqueries, "sub_section_types": section_types,
        "llm_calls": state["llm_calls"],  # run_task() đã mutate state["llm_calls"] in-place
    }


def fan_out(state: State) -> list[Send]:
    subqueries = state.get("subqueries") or []
    section_types = state.get("sub_section_types") or []
    return [
        Send("retrieve_sub", {"query": q, "section_type": t})
        for q, t in zip(subqueries, section_types)
    ]


async def retrieve_sub(payload: dict) -> dict:
    query = payload["query"]
    section_type = payload.get("section_type")
    candidates = retrieve(query, section_type=section_type)
    tagged = [{**c, "_subquery": query} for c in candidates[:10]]
    return {"sub_results": tagged}


async def merge_evidence(state: State) -> dict:
    by_subquery: dict[str, list[dict]] = {}
    for c in state.get("sub_results") or []:
        by_subquery.setdefault(c.get("_subquery", ""), []).append(c)

    # giữ tối đa 3 chunk/subquery (mục 8.4)
    capped = {k: v[:MAX_CHUNKS_PER_SUBQUERY] for k, v in by_subquery.items()}

    pool: dict[str, dict] = dict(state.get("retrieved_pool") or {})
    for chunks in capped.values():
        for c in chunks:
            pool.setdefault(c["chunk_id"], c)

    # rerank CÓ đảm bảo mỗi subquery giữ ít nhất 1 chunk — tránh 1 bên bị đè mất
    # hoàn toàn khi so sánh 2 thực thể (xem phát hiện ở eval/run_routing_compare.py)
    chunks = rerank_balanced_by_group(state["standalone_question"], capped)
    return {"retrieved_pool": pool, "reranked_chunks": chunks}
