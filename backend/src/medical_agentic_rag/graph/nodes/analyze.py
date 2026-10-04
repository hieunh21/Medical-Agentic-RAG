"""Node analyze_query — phân loại câu hỏi, tách thực thể/khía cạnh (mục 8.2).

safety_label là lớp 2 của safety, hợp nhất với lớp 1 (safety_rules) theo mức nghiêm trọng;
emergency / self_harm dừng pipeline ở route_by_type (mục 10.1), trừ emergency mà LLM xác định là
câu hỏi kiến thức chung (situation="general") — xem safety/rules.py và graph/nodes/safety.py.
"""
from __future__ import annotations

from medical_agentic_rag.budget import degrade_on_llm_error
from medical_agentic_rag.graph.state import DEFAULT_ASPECT, State
from medical_agentic_rag.llm.client import run_task
from medical_agentic_rag.llm.prompts import build_analyze_prompt
from medical_agentic_rag.llm.schemas import QueryAnalysis
from medical_agentic_rag.safety.rules import more_severe


# Không phân loại được -> question_type rỗng -> route_by_type rơi về hybrid_retrieve.
# safety_unverified: lớp 2 (LLM) không chạy được nên safety chưa được kiểm tra đầy đủ -> validate.py chèn khuyến cáo 115.
@degrade_on_llm_error(lambda s: {"question_type": "", "aspects": [DEFAULT_ASPECT], "safety_unverified": True})
async def analyze_query(state: State) -> dict:
    prompt = build_analyze_prompt(state["standalone_question"])
    analysis: QueryAnalysis = await run_task("analyze_query", prompt, state, schema=QueryAnalysis)
    label = more_severe(state.get("safety_label"), analysis.safety_label)  # nghiêm trọng hơn giữa 2 lớp
    # Chỉ hạ emergency (không bao giờ hạ self_harm) khi LLM nói rõ đây là câu hỏi kiến thức chung; hạ rồi
    # vẫn trả lời nhưng chèn khuyến cáo gọi 115 (xem validate.py). Mơ hồ / lỗi LLM -> giữ nguyên, tức là dừng.
    emergency_topic = label == "emergency" and analysis.situation == "general"
    if emergency_topic:
        label = "normal"
    return {
        "question_type": analysis.question_type,
        "safety_label": label,
        "emergency_topic": emergency_topic,
        "entities": analysis.entities,
        "aspects": analysis.aspects or [DEFAULT_ASPECT],
        "target_section_types": analysis.target_section_types,
        "llm_calls": state["llm_calls"],  # run_task() đã mutate state["llm_calls"] in-place
    }
