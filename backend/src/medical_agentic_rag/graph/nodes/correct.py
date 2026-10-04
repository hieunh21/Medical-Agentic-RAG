"""Node sửa evidence: targeted_retrieve (còn thiếu khía cạnh) và rewrite_query (không có gì liên quan).

Mục 7.3. Việc map khía cạnh -> section_type và tách "thực thể chính" làm đầy đủ hơn ở
Phase 3 (analyze_query); ở Phase 2 chỉ có một khía cạnh mặc định nên query ghép thẳng
câu hỏi + khía cạnh là đủ dùng.
"""
from __future__ import annotations

import asyncio

from medical_agentic_rag.budget import degrade_on_llm_error
from medical_agentic_rag.graph.state import State
from medical_agentic_rag.llm.client import run_task
from medical_agentic_rag.llm.prompts import build_rewrite_prompt
from medical_agentic_rag.llm.schemas import Rewrites
from medical_agentic_rag.retrieval.hybrid import retrieve
from medical_agentic_rag.retrieval.rerank import rerank_and_select

ASPECT_TO_SECTION_TYPE = {
    "khi nào đi khám": "when_to_see_doctor",
    "triệu chứng": "symptom",
    "nguyên nhân": "cause_risk",
    "chẩn đoán": "diagnosis",
    "điều trị": "treatment",
    "phòng ngừa": "prevention",
    "biến chứng": "complication_prognosis",
}


def _targeted_retrieve_sync(state: State) -> tuple[dict, list[dict]]:
    pool = dict(state["retrieved_pool"])
    for aspect in state["missing_aspects"]:
        query = f"{state['standalone_question']} {aspect}"
        section_type = ASPECT_TO_SECTION_TYPE.get(aspect)
        for c in retrieve(query, section_type=section_type):
            pool.setdefault(c["chunk_id"], c)
    return pool, rerank_and_select(state["standalone_question"], list(pool.values()))


async def targeted_retrieve(state: State) -> dict:
    pool, chunks = await asyncio.to_thread(_targeted_retrieve_sync, state)
    return {
        "retrieved_pool": pool,
        "reranked_chunks": chunks,
        "corrections": state["corrections"] + 1,
    }


def _rewrite_retrieve_sync(state: State, queries: list[str]) -> tuple[dict, list[dict]]:
    pool: dict[str, dict] = {}
    for q in queries[:3]:
        for c in retrieve(q):
            pool.setdefault(c["chunk_id"], c)
    if not pool:  # không sinh được truy vấn nào hữu ích — giữ pool cũ thay vì xoá sạch
        pool = dict(state["retrieved_pool"])
    return pool, rerank_and_select(state["standalone_question"], list(pool.values()))


# Không viết lại được: giữ nguyên evidence hiện có, vẫn tính 1 lượt sửa để vòng lặp kết thúc.
@degrade_on_llm_error(lambda s: {"corrections": s["corrections"] + 1})
async def rewrite_query(state: State) -> dict:
    prompt = build_rewrite_prompt(state["standalone_question"])
    rewrites: Rewrites = await run_task("rewrite_query", prompt, state, schema=Rewrites)

    pool, chunks = await asyncio.to_thread(_rewrite_retrieve_sync, state, rewrites.queries)
    return {
        "retrieved_pool": pool,
        "reranked_chunks": chunks,
        "corrections": state["corrections"] + 1,
        "llm_calls": state["llm_calls"],  # run_task() đã mutate state["llm_calls"] in-place
    }
