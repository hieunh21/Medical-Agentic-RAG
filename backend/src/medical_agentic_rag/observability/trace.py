"""Log mỗi lời gọi LLM ra JSONL — để tính chi phí / debug (mục 7.5, 10.4)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

TRACE_LOG = os.getenv("TRACE_LOG", "data/traces.jsonl")


def log_llm(state: dict, task: str, attempt: int, t0: float, ok: bool, error: str | None = None) -> None:
    row = {
        "ts": time.time(),
        "trace_id": state.get("trace_id"),
        "task": task,
        "attempt": attempt,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "ok": ok,
        "error": error,
    }
    path = Path(TRACE_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
