"""research_agent — LLM tự chọn tool trong ngân sách cứng (mục 8.5).

Agent không viết câu trả lời, chỉ chọn evidence; danh sách chunk chọn ra vẫn đi
qua rerank -> grade -> generate như mọi route khác (xem graph/build.py).
"""
from __future__ import annotations

import time

from google.genai import types

from ingestion.qdrant_store import get_chunks_by_ids, get_client
from medical_agentic_rag.agent.tools import AGENT_TOOLS, execute_tool
from medical_agentic_rag.config import settings
from medical_agentic_rag.graph.state import State
from medical_agentic_rag.llm.client import run_task

AGENT_PROMPT_TEMPLATE = """Bạn là agent tìm kiếm thông tin y khoa trên YouMed để trả lời câu hỏi sau.
Bạn CHỈ được chọn evidence, KHÔNG viết câu trả lời cuối cùng.

CÂU HỎI: {question}
THỰC THỂ liên quan: {entities}

Dùng các tool để tìm bài, đọc nội dung, rồi gọi "finish" với danh sách chunk_id tốt nhất
(ưu tiên đọc nguyên văn bằng read_chunks trước khi đưa chunk_id vào finish). Ngân sách tối
đa {budget} lượt gọi tool.

QUAN TRỌNG — câu hỏi nối nhiều thực thể (vd "A có gây ra B không", "A và B liên quan
gì"): PHẢI tìm kiếm riêng cho TỪNG thực thể (mỗi thực thể ít nhất 1 lần find_article/
search_youmed trên TOÀN corpus, không giới hạn article_id của thực thể khác), rồi mới
được finish. Một chunk chỉ vì nhắc tới cả 2 từ khoá KHÔNG có nghĩa là đã đủ bằng chứng
cho cả 2 phía — đọc kỹ chiều quan hệ (A gây B, hay B gây A) trước khi finish.

Khi finish: đưa vào chunk_ids TẤT CẢ chunk liên quan đã đọc được cho MỖI thực thể (đọc
bằng read_chunks trước), không chỉ 1 chunk duy nhất — thiếu bằng chứng cho 1 thực thể
thì câu trả lời sau này sẽ không đầy đủ."""


def _agent_prompt(question: str, entities: list[str], budget_calls: int) -> str:
    return AGENT_PROMPT_TEMPLATE.format(
        question=question, entities=", ".join(entities) or "(không rõ)", budget=budget_calls,
    )


def _first_function_call(resp):
    calls = resp.function_calls
    return calls[0] if calls else None


def _collect_chunk_ids(result) -> list[str]:
    items = result if isinstance(result, list) else [result]
    return [it["chunk_id"] for it in items if isinstance(it, dict) and it.get("chunk_id")]


async def research_agent(state: State) -> dict:
    seen_ids: set[str] = set()
    chosen_ids: list[str] = []
    calls = 0
    t0 = time.perf_counter()

    contents: list = [_agent_prompt(
        state["standalone_question"], state.get("entities") or [], settings.AGENT_MAX_TOOL_CALLS,
    )]

    empty_retries = 0
    while calls < settings.AGENT_MAX_TOOL_CALLS and (time.perf_counter() - t0) < settings.AGENT_TIMEOUT_S:
        resp = await run_task("research_agent", contents, state, tools=AGENT_TOOLS)
        call = _first_function_call(resp)

        # Model thỉnh thoảng trả rỗng (không function_call, không text) dù còn ngân
        # sách — coi là trục trặc tạm thời, thử lại cùng lượt 1 lần thay vì kết luận
        # agent chủ động dừng (không tính vào ngân sách tool call).
        if call is None and not (resp.text or "").strip() and empty_retries < 1:
            empty_retries += 1
            continue

        if call is None or call.name == "finish":
            chosen_ids = list((call.args or {}).get("chunk_ids", [])) if call else []
            break

        result = await execute_tool(call.name, call.args or {})
        calls += 1
        seen_ids.update(_collect_chunk_ids(result))

        contents.append(resp.candidates[0].content)
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_function_response(
                name=call.name,
                response={"result": result, "remaining_tool_calls": settings.AGENT_MAX_TOOL_CALLS - calls},
            )],
        ))

    # Chỉ nhận chunk_id agent thật sự đã đọc được qua tool (chặn hallucination);
    # hết ngân sách mà chưa finish -> dùng toàn bộ chunk đã thấy.
    final_ids = [i for i in chosen_ids if i in seen_ids] or list(seen_ids)
    client = get_client()
    chunks = get_chunks_by_ids(client, final_ids[: settings.AGENT_MAX_POOL]) if final_ids else []

    return {
        "retrieved_pool": {c["chunk_id"]: c for c in chunks},
        "agent_tool_calls": calls,
        "llm_calls": state["llm_calls"],  # run_task() đã mutate state["llm_calls"] in-place
    }
