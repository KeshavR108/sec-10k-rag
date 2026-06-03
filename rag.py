import os
from typing import Optional

import anthropic
from dotenv import load_dotenv
from embedding import query_chunks

load_dotenv()

_api_key = os.environ.get("ANTHROPIC_API_KEY")
_client = None
if _api_key:
    try:
        _client = anthropic.Anthropic(api_key=_api_key)
    except Exception:
        _client = None

_MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """You are a financial analyst assistant. Answer the user's question using ONLY the source excerpts provided below. Don't use outside knowledge.

Rules:
- If the answer is not present in the sources, say "The provided filings do not contain enough information to answer this question."
- Always cite your sources using [SOURCE N] inline.
- After your answer, include "Citations" section listing each source you used with: Source N → Company | Section.
- Be concise and factual."""


def _build_context(chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[SOURCE {i}]\n"
            f"Company: {chunk['company']}\n"
            f"Section: {chunk['section']}\n"
            f"Text: {chunk['text']}"
        )
    return "\n\n".join(context_blocks)


def answer_question(
    question: str,
    top_k: int = 5,
    company_filter: Optional[str] = None,
    section_filter: Optional[str] = None,
    doc_id_filter: Optional[str] = None,
    persist_dir: str = "./chroma_db",
) -> dict:
    try:
        chunks = query_chunks(
            question=question,
            top_k=top_k,
            doc_id=doc_id_filter,
            company=company_filter,
            section=section_filter,
            persist_dir=persist_dir,
        )
    except Exception as exc:
        return {
            "question": question,
            "answer": None,
            "sources": [],
            "truncated": False,
            "error": f"Retrieval failed: {exc}",
        }

    if not chunks:
        return {
            "question": question,
            "answer": "No relevant documents found. Please ingest some 10-K filings first.",
            "sources": [],
            "truncated": False,
            "error": None,
        }

    context = _build_context(chunks)
    user_message = f"{context}\n\nQuestion: {question}"

    if _client is None:
        return {
            "question": question,
            "answer": None,
            "sources": [
                {
                    "source_index": i + 1,
                    "company": c["company"],
                    "section": c["section"],
                    "doc_id": c["doc_id"],
                    "excerpt": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
                    "relevance_distance": c["distance"],
                }
                for i, c in enumerate(chunks)
            ],
            "truncated": False,
            "error": "API key is not set. Returned retrieved sources without a generated answer.",
        }

    try:
        message = _client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIError as exc:
        return {
            "question": question,
            "answer": None,
            "sources": [],
            "truncated": False,
            "error": f"Claude API error: {exc}",
        }

    text_blocks = [b for b in message.content if b.type == "text"]
    if not text_blocks:
        return {
            "question": question,
            "answer": None,
            "sources": [],
            "truncated": False,
            "error": "Claude returned no text content.",
        }

    answer_text = text_blocks[0].text.strip()

    truncated = message.stop_reason == "max_tokens"
    if truncated:
        answer_text += "\n\n[Response truncated — consider increasing max_tokens]"

    sources = [
        {
            "source_index": i + 1,
            "company": c["company"],
            "section": c["section"],
            "doc_id": c["doc_id"],
            "excerpt": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
            "relevance_distance": c["distance"],
        }
        for i, c in enumerate(chunks)
    ]

    return {
        "question": question,
        "answer": answer_text,
        "sources": sources,
        "truncated": truncated,
        "error": None,
    }
