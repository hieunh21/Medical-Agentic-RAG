"""Node sửa evidence: targeted_retrieve (còn thiếu khía cạnh) và rewrite_query (không có gì liên quan).

Mục 7.3. Việc map khía cạnh -> section_type và tách "thực thể chính" làm đầy đủ hơn ở
Phase 3 (analyze_query); ở Phase 2 chỉ có một khía cạnh mặc định nên query ghép thẳng
câu hỏi + khía cạnh là đủ dùng.
"""
from __future__ import annotations

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


async def targeted_retrieve(state: State) -> dict:
    pool = dict(state["retrieved_pool"])
    for aspect in state["missing_aspects"]:
        query = f"{state['standalone_question']} {aspect}"
        section_type = ASPECT_TO_SECTION_TYPE.get(aspect)
        for c in retrieve(query, section_type=section_type):
            pool.setdefault(c["chunk_id"], c)

    chunks = rerank_and_select(state["standalone_question"], list(pool.values()))
    return {
        "retrieved_pool": pool,
        "reranked_chunks": chunks,
        "corrections": state["corrections"] + 1,
    }


async def rewrite_query(state: State) -> dict:
    prompt = build_rewrite_prompt(state["standalone_question"])
    rewrites: Rewrites = await run_task("rewrite_query", prompt, state, schema=Rewrites)

    pool: dict[str, dict] = {}
    for q in rewrites.queries[:3]:
        for c in retrieve(q):
            pool.setdefault(c["chunk_id"], c)
    if not pool:  # không sinh được truy vấn nào hữu ích — giữ pool cũ thay vì xoá sạch
        pool = dict(state["retrieved_pool"])

    chunks = rerank_and_select(state["standalone_question"], list(pool.values()))
    return {
        "retrieved_pool": pool,
        "reranked_chunks": chunks,
        "corrections": state["corrections"] + 1,
        "llm_calls": state["llm_calls"],  # run_task() đã mutate state["llm_calls"] in-place
    }
