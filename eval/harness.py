"""Chạy hệ thống trên một câu hỏi và chấm câu trả lời — dùng chung cho các script trong eval/.

Tách riêng khỏi script báo cáo vì phần "gọi graph, ghi lại kết quả, chấm trích dẫn" giống nhau
cho mọi lượt đánh giá; chỉ cách tổng hợp thành bảng là khác.

Import module này sẽ đăng ký task "eval_judge": judge mặc định dùng chung model với
LLM_MODEL_JUDGE, đặt EVAL_JUDGE_MODEL để dùng model khác (giảm thiên vị tự chấm).
"""
from __future__ import annotations

import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

from medical_agentic_rag.citation import validator as v  # noqa: E402
from medical_agentic_rag.graph.nodes.validate import FILTERED_NOTE  # noqa: E402
from medical_agentic_rag.graph.state import init_state  # noqa: E402
from medical_agentic_rag.llm import client as llm_client  # noqa: E402
from medical_agentic_rag.llm.client import run_task  # noqa: E402
from medical_agentic_rag.llm.prompts import build_verify_claims_prompt  # noqa: E402
from medical_agentic_rag.llm.schemas import ClaimVerdicts  # noqa: E402
from medical_agentic_rag.llm.tasks import TASKS, TaskSpec  # noqa: E402
from medical_agentic_rag.safety.templates import EMERGENCY_TOPIC_NOTE, HIGH_RISK_PREFIX  # noqa: E402

TIERS = ["easy", "medium", "hard"]

# Câu trả lời mang nghĩa "từ chối / chưa có thông tin" (template no_info, out_of_scope, hoặc LLM tự nói).
REFUSAL_RE = re.compile(
    r"chưa có (?:thông tin|bài)|ngoài phạm vi|không có thông tin|không đủ thông tin", re.IGNORECASE)

TASKS["eval_judge"] = TaskSpec("eval_judge", 0.0, 2048)
llm_client.MODEL_BY_TIER["eval_judge"] = os.getenv("EVAL_JUDGE_MODEL") or llm_client.MODEL_BY_TIER["judge"]


async def judge_answer(answer: str, chunks: list[dict]) -> Optional[dict]:
    """Faithfulness + citation precision bằng LLM-judge, dùng lại tách câu / bước 1 của validator.

    Trả None nếu không có khẳng định y khoa nào để chấm (câu từ chối, template safety).
    """
    # câu template do code chèn (khuyến cáo high-risk / cấp cứu, ghi chú lược bỏ) không cần nguồn
    for template in (HIGH_RISK_PREFIX, EMERGENCY_TOPIC_NOTE, FILTERED_NOTE):
        answer = answer.replace(template.strip(), "")
    answer = answer.strip()
    if not answer or REFUSAL_RE.search(answer) and "[" not in answer:
        return None

    sentences = v.split_sentences(answer)
    v.check_citations(sentences, len(chunks))
    claims = [s for s in sentences if s.needs_cite]
    if not claims:
        return None

    targets = v.to_verify(sentences)
    if targets:
        prompt = build_verify_claims_prompt([(s.id, s.text, s.refs) for s in targets], chunks)
        try:
            res: ClaimVerdicts = await run_task(
                "eval_judge", prompt, {"llm_calls": 0, "trace_id": "eval-judge"}, schema=ClaimVerdicts)
            v.apply_verdicts(sentences, {c.sentence_id: (c.verdict, c.reason) for c in res.verdicts})
        except Exception:  # noqa: BLE001 — judge lỗi: bỏ câu này, không làm hỏng cả lượt chạy
            return {"judge_error": True}

    cited = [s for s in claims if s.refs]
    prec = sum(1.0 for s in cited if s.verdict == "supported") + sum(0.5 for s in cited if s.verdict == "partial")
    return {
        "n_claims": len(claims), "unsupported": sum(s.verdict == "unsupported" for s in claims),
        "n_cited": len(cited), "n_uncited": len(claims) - len(cited), "cite_score": prec,
        "counts": v.claim_counts(sentences),
    }


async def run_one(graph, q: dict, with_judge: bool) -> dict:
    """Chạy 1 câu hỏi qua graph, trả về 1 dòng kết quả để cache ra JSONL."""
    row = {"id": q["id"], "group": q["group"], "difficulty": q.get("difficulty")}
    t0 = time.perf_counter()
    try:
        result = await graph.ainvoke(init_state(q["question"], trace_id=str(uuid.uuid4())))
    except Exception as exc:  # noqa: BLE001 — 1 câu lỗi không được làm dừng cả lượt chạy
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return row

    row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    chunks = result.get("reranked_chunks") or []
    answer = result.get("final_answer", "")
    urls: list[str] = []
    for c in chunks:
        if c["url"] not in urls:
            urls.append(c["url"])
    row.update(
        answer=answer, urls=urls[:5], chunk_ids=[c["chunk_id"] for c in chunks],
        llm_calls=result.get("llm_calls", 0), safety_label=result.get("safety_label"),
        evidence_status=result.get("evidence_status"), question_type=result.get("question_type"),
        refused=bool(REFUSAL_RE.search(answer)) and not re.search(r"\[\d+\]", answer),
    )
    if q["group"] == "H" and with_judge:
        row["judge"] = await judge_answer(answer, chunks)
    return row
