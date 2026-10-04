"""Node trả lời: generate_answer (có evidence, ra bản nháp) và các template không LLM.

generate_answer chỉ tạo `draft_answer`; citation_validator (graph/nodes/validate.py) mới chốt
final_answer + messages. Khi validator yêu cầu viết lại, generate_answer chạy lại với danh
sách câu bị loại.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from medical_agentic_rag.answer import build_prompt, format_sources
from medical_agentic_rag.budget import BudgetExceeded
from medical_agentic_rag.config import settings
from medical_agentic_rag.errors import LLMTaskFailed
from medical_agentic_rag.graph.state import State
from medical_agentic_rag.llm.client import run_task
from medical_agentic_rag.safety import templates

GENERATE_FAILED = "Hệ thống tạm thời chưa tạo được câu trả lời. Bạn có thể tham khảo các bài viết liên quan bên dưới hoặc thử lại sau."
NO_INFO = "YouMed hiện chưa có bài viết về vấn đề này. Bạn nên hỏi ý kiến bác sĩ để được tư vấn chính xác."


async def generate_answer_node(state: State, config: RunnableConfig) -> dict:
    """config["configurable"]["on_token"] (nếu có) nhận từng mảnh text để API stream ra frontend.

    Lượt viết lại (regenerations > 0) KHÔNG stream: bản nháp cũ đã hiện trên UI rồi, stream tiếp
    sẽ chèn chữ vào giữa câu trả lời cũ. API tự thay toàn bộ text khi nhận event done.
    """
    chunks = state["reranked_chunks"]
    if "coverage" not in state:  # chưa qua grader (baseline / fast path): không khẳng định đủ hay thiếu
        status = None
    else:
        status = "sufficient" if state["coverage"] >= settings.COVERAGE_THRESHOLD else "partial"
    prompt = build_prompt(
        state["standalone_question"], chunks,
        evidence_status=status, question_type=state.get("question_type"),
        safety_label=state.get("safety_label"), rejected=state.get("rejected_sentences") or None,
    )
    out = {
        "regenerate": False,
        "sources": format_sources(chunks),
        "evidence_status": status,
        "active_article_ids": list({c["article_id"] for c in chunks}),
    }
    on_token = (config.get("configurable") or {}).get("on_token") if not state.get("regenerations") else None
    try:
        out["draft_answer"] = await run_task("generate_answer", prompt, state, on_token=on_token)
        out["generate_failed"] = False
    except (LLMTaskFailed, BudgetExceeded):
        if state.get("draft_answer") and state.get("regenerations"):
            out["draft_answer"] = state["draft_answer"]  # viết lại lỗi: giữ bản nháp cũ, validator lọc tiếp
        else:
            out.update(draft_answer=GENERATE_FAILED, generate_failed=True, evidence_status="insufficient")
    out["llm_calls"] = state["llm_calls"]  # run_task() đã mutate state["llm_calls"] in-place
    return out


def _template_response(text: str) -> dict:
    return {
        "final_answer": text, "sources": [], "evidence_status": "insufficient",
        "messages": [AIMessage(content=text)],
    }


def no_info_response_node(state: State) -> dict:
    return _template_response(NO_INFO)


OUT_OF_SCOPE = "Câu hỏi này nằm ngoài phạm vi thông tin y khoa mà YouMed hỗ trợ. Bạn thử hỏi về triệu chứng, bệnh lý hoặc cách chăm sóc sức khoẻ nhé."


def out_of_scope_response_node(state: State) -> dict:
    return _template_response(OUT_OF_SCOPE)


def safety_response_node(state: State) -> dict:
    """emergency / self_harm: template cố định đã duyệt, không LLM, không retrieval."""
    label = state.get("safety_label")
    text = templates.self_harm_response() if label == "self_harm" else templates.emergency_response()
    return _template_response(text)
