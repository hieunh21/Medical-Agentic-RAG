#!/usr/bin/env python3
"""Local embed + rerank server cho retrieval.

    POST /embed   {texts, task: "query"|"document", titles?}
    POST /rerank  {query, documents}

Usage:
    uvicorn model_server.server:app --port 8001
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional

import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv(Path(__file__).parent / ".env")

EMBED_MODEL = os.getenv("EMBED_MODEL", "google/embeddinggemma-300m")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
DEVICE = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

# bfloat16 trên CUDA: GPU 4GB VRAM không đủ chỗ cho cả 2 model + batch activation ở fp32.
# Dùng bfloat16 (không phải float16) vì họ Gemma dễ tràn số / ra NaN ở float16.
_dtype_kwargs = {"model_kwargs": {"torch_dtype": torch.bfloat16}} if DEVICE == "cuda" else {}

app = FastAPI(title="medical-agentic-rag model server")
model = SentenceTransformer(EMBED_MODEL, device=DEVICE, **_dtype_kwargs)
reranker = CrossEncoder(RERANK_MODEL, device=DEVICE, **_dtype_kwargs)


class EmbedReq(BaseModel):
    texts: List[str]
    task: Literal["query", "document"]
    titles: Optional[List[str]] = None


class EmbedResp(BaseModel):
    embeddings: List[List[float]]
    dim: int


class RerankReq(BaseModel):
    query: str
    documents: List[str]


class RerankResp(BaseModel):
    scores: List[float]


@app.post("/embed", response_model=EmbedResp)
def embed(req: EmbedReq) -> EmbedResp:
    if req.task == "query":
        vecs = model.encode_query(req.texts, batch_size=32, normalize_embeddings=True)
    else:
        titles = req.titles or ["none"] * len(req.texts)
        docs = [f"title: {t} | text: {x}" for t, x in zip(titles, req.texts)]
        vecs = model.encode(docs, prompt="", batch_size=16, normalize_embeddings=True)
    return EmbedResp(embeddings=vecs.tolist(), dim=int(vecs.shape[1]))


@app.post("/rerank", response_model=RerankResp)
def rerank(req: RerankReq) -> RerankResp:
    scores = reranker.predict([(req.query, d) for d in req.documents], batch_size=16)
    return RerankResp(scores=[float(s) for s in scores])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": DEVICE, "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL}
