"""Embed + rerank trên Hugging Face ZeroGPU — thay cho model_server/ khi deploy.

API giữ nguyên ý nghĩa của model_server/server.py, chỉ đổi cách gọi (gradio_client thay vì HTTP POST):
    /embed   payload {"texts": [...], "task": "query" | "document", "titles"?: [...]}
    /rerank  payload {"query": "...", "documents": [...]}
    /ping    không dùng GPU — để đánh thức Space và kiểm tra còn sống

Mỗi kết quả trả kèm "gpu_s": thời gian chạy trong hàm @spaces.GPU, xấp xỉ phần quota ZeroGPU bị trừ.
"""
from __future__ import annotations

import json
import os
import time

import spaces  # có sẵn trên ZeroGPU Space
import gradio as gr
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL = os.getenv("EMBED_MODEL", "google/embeddinggemma-300m")  # gated: cần secret HF_TOKEN
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# ZeroGPU: model phải đặt lên "cuda" ở cấp module (ngoài @spaces.GPU là chế độ giả lập CUDA).
# bfloat16 như model_server/server.py — họ Gemma dễ ra NaN ở float16.
_dtype = {"model_kwargs": {"torch_dtype": torch.bfloat16}}
embedder = SentenceTransformer(EMBED_MODEL, device="cuda", **_dtype)
reranker = CrossEncoder(RERANK_MODEL, device="cuda", **_dtype)


@spaces.GPU(duration=20)
def _embed(texts: list[str], task: str, titles: list[str] | None) -> tuple[list[list[float]], float]:
    t0 = time.perf_counter()
    if task == "query":
        vecs = embedder.encode_query(texts, batch_size=32, normalize_embeddings=True)
    else:
        titles = titles or ["none"] * len(texts)
        docs = [f"title: {t} | text: {x}" for t, x in zip(titles, texts)]
        vecs = embedder.encode(docs, prompt="", batch_size=16, normalize_embeddings=True)
    return vecs.tolist(), time.perf_counter() - t0


@spaces.GPU(duration=30)
def _rerank(query: str, documents: list[str]) -> tuple[list[float], float]:
    t0 = time.perf_counter()
    scores = reranker.predict([(query, d) for d in documents], batch_size=16)
    return [float(s) for s in scores], time.perf_counter() - t0


def embed(payload: str) -> str:
    req = json.loads(payload)
    vecs, gpu_s = _embed(req["texts"], req.get("task", "query"), req.get("titles"))
    return json.dumps({"embeddings": vecs, "dim": len(vecs[0]) if vecs else 0, "gpu_s": round(gpu_s, 3)})


def rerank(payload: str) -> str:
    req = json.loads(payload)
    if not req.get("documents"):
        return json.dumps({"scores": [], "gpu_s": 0.0})
    scores, gpu_s = _rerank(req["query"], req["documents"])
    return json.dumps({"scores": scores, "gpu_s": round(gpu_s, 3)})


def ping(_: str = "") -> str:
    return json.dumps({"status": "ok", "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL})


with gr.Blocks(title="MedRAG models") as demo:
    gr.Markdown(
        "API embed + rerank cho Medical Agentic RAG, chạy trên ZeroGPU. "
        "Không phải giao diện cho người dùng — backend gọi qua `gradio_client`."
    )
    payload = gr.Textbox(label="payload (JSON)")
    result = gr.Textbox(label="result (JSON)")
    gr.Button("embed").click(embed, payload, result, api_name="embed")
    gr.Button("rerank").click(rerank, payload, result, api_name="rerank")
    gr.Button("ping").click(ping, payload, result, api_name="ping")

demo.queue(default_concurrency_limit=4).launch()
