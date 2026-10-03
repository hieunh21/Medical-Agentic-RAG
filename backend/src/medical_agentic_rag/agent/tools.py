"""6 tool của research agent (mục 8.5) — code thường, agent chỉ được gọi qua allowlist."""
from __future__ import annotations

from typing import Any, Optional

from google.genai import types

from ingestion.qdrant_store import get_chunks_by_article, get_chunks_by_ids, get_client
from medical_agentic_rag.retrieval.article_index import find_articles, get_related_articles
from medical_agentic_rag.retrieval.hybrid import retrieve

SNIPPET_LEN = 200
MAX_READ_CHUNKS = 4

TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="find_article",
        description="Tìm bài YouMed theo tên bệnh/triệu chứng (so khớp mờ, không cần viết đúng chính tả).",
        parameters_json_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    ),
    types.FunctionDeclaration(
        name="get_article_outline",
        description="Lấy mục lục (danh sách section) của 1 bài YouMed theo article_id.",
        parameters_json_schema={
            "type": "object",
            "properties": {"article_id": {"type": "string"}},
            "required": ["article_id"],
        },
    ),
    types.FunctionDeclaration(
        name="search_youmed",
        description="Tìm kiếm hybrid trên toàn bộ corpus YouMed, trả về đoạn trích ngắn (snippet).",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "section_type": {"type": "string"},
                "article_id": {"type": "string"},
                "k": {"type": "integer"},
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="read_chunks",
        description="Đọc nguyên văn tối đa 4 chunk theo chunk_id (dùng sau khi đã tìm bằng search_youmed).",
        parameters_json_schema={
            "type": "object",
            "properties": {"chunk_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["chunk_ids"],
        },
    ),
    types.FunctionDeclaration(
        name="related_articles",
        description="Lấy danh sách bài liên quan tới 1 article_id.",
        parameters_json_schema={
            "type": "object",
            "properties": {"article_id": {"type": "string"}},
            "required": ["article_id"],
        },
    ),
    types.FunctionDeclaration(
        name="finish",
        description="Kết thúc tìm kiếm, chọn ra các chunk_id tốt nhất làm evidence để trả lời câu hỏi.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "chunk_ids": {"type": "array", "items": {"type": "string"}},
                "note": {"type": "string"},
            },
            "required": ["chunk_ids"],
        },
    ),
]

AGENT_TOOLS = [types.Tool(function_declarations=TOOL_DECLARATIONS)]
ALLOWLIST = {d.name for d in TOOL_DECLARATIONS}


def _tool_find_article(name: str) -> list[dict]:
    return find_articles(name, k=5)


def _tool_get_article_outline(article_id: str) -> dict:
    client = get_client()
    chunks = get_chunks_by_article(client, article_id)
    if not chunks:
        return {"error": f"không tìm thấy bài {article_id}"}
    chunks.sort(key=lambda c: c.get("chunk_index", 0))
    return {
        "title": chunks[0].get("title"),
        "updated_date": chunks[0].get("updated_date"),
        "sections": [
            {"chunk_id": c["chunk_id"], "section_path": c.get("section_path"), "section_type": c.get("section_type")}
            for c in chunks
        ],
    }


def _tool_search_youmed(
    query: str, section_type: Optional[str] = None, article_id: Optional[str] = None, k: int = 5,
) -> list[dict]:
    candidates = retrieve(query, limit=max(k, 1) * 3, section_type=section_type, article_id=article_id)
    out = []
    for c in candidates[: min(k, 5)]:
        out.append({
            "chunk_id": c["chunk_id"], "title": c["title"], "section_path": c.get("section_path"),
            "snippet": c["text"][:SNIPPET_LEN],
        })
    return out


def _tool_read_chunks(chunk_ids: list[str]) -> list[dict]:
    client = get_client()
    chunks = get_chunks_by_ids(client, chunk_ids[:MAX_READ_CHUNKS])
    return [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks]


def _tool_related_articles(article_id: str) -> list[dict]:
    return get_related_articles(article_id)


DISPATCH = {
    "find_article": _tool_find_article,
    "get_article_outline": _tool_get_article_outline,
    "search_youmed": _tool_search_youmed,
    "read_chunks": _tool_read_chunks,
    "related_articles": _tool_related_articles,
}


async def execute_tool(name: str, args: dict[str, Any]) -> Any:
    if name not in ALLOWLIST or name not in DISPATCH:
        return {"error": f"tool '{name}' không nằm trong allowlist"}
    try:
        return DISPATCH[name](**args)
    except TypeError as exc:
        return {"error": f"tham số sai schema: {exc}"}
