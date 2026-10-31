---
title: MedRAG Models
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.27.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Embed + rerank API (ZeroGPU) cho Medical Agentic RAG
---

Embed (`google/embeddinggemma-300m`) và rerank (`BAAI/bge-reranker-v2-m3`) cho
[Medical Agentic RAG](https://github.com/hieunh21/Medical-Agentic-RAG), chạy trên ZeroGPU.

Cần secret `HF_TOKEN` (token chỉ đọc, đã chấp nhận giấy phép Gemma) để tải model embed.
