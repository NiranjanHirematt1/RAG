"""Core RAG logic: ingest PDFs and answer questions with Groq.

This module is framework-agnostic – it is used by both the FastAPI/Inngest
service (main.py) and the Streamlit UI (app.py).
"""
import os
import uuid

from dotenv import load_dotenv
from groq import Groq

from data_loader import load_and_chunk_pdf, embed_texts
from qdrant_db import QdrantStorage
from custom_types import (
    RAGChunkAndSrc,
    RAGUpsertResult,
    RAGSearchResult,
    RAGQueryResult,
)

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = (
    "You answer questions based only on the context provided. "
    "If the context does not contain the answer, respond with "
    "\"I don't know based on the provided document.\""
)

# Reuse a single vector store / client instance per process.
_storage: QdrantStorage | None = None
_groq_client: Groq | None = None


def get_storage() -> QdrantStorage:
    global _storage
    if _storage is None:
        _storage = QdrantStorage()
    return _storage


def get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in the environment/.env")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def ingest_pdf(pdf_path: str, source_id: str | None = None) -> RAGUpsertResult:
    """Load a PDF, chunk it, embed it and upsert the vectors into Qdrant."""
    source_id = source_id or os.path.basename(pdf_path)

    chunks = load_and_chunk_pdf(pdf_path)
    chunk_and_src = RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    if not chunk_and_src.chunks:
        return RAGUpsertResult(ingested=0)

    vecs = embed_texts(chunk_and_src.chunks)
    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}_{i}"))
        for i in range(len(chunk_and_src.chunks))
    ]
    payloads = [
        {"text": chunk, "source": source_id}
        for chunk in chunk_and_src.chunks
    ]

    storage = get_storage()
    # Remove old chunks first, so re-uploading an edited PDF leaves no stale text.
    storage.delete_source(source_id)
    storage.upsert(ids, vecs, payloads)
    return RAGUpsertResult(ingested=len(chunk_and_src.chunks))


# --------------------------------------------------------------------------- #
# Retrieval + generation
# --------------------------------------------------------------------------- #
def search(question: str, top_k: int = 5) -> RAGSearchResult:
    query_vec = embed_texts([question])[0]
    found = get_storage().search(query_vec, top_k)
    return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])


def answer_question(question: str, top_k: int = 5) -> RAGQueryResult:
    found = search(question, top_k)

    if not found.contexts:
        return RAGQueryResult(
            answer="I don't know based on the provided document. "
            "Try ingesting a PDF first.",
            sources=found.sources,
            num_contexts=0,
        )

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the context below to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using only the context above."
    )

    client = get_groq()
    res = client.chat.completions.create(
        model=GROQ_MODEL,
        # gpt-oss models spend tokens on reasoning first; 500 could truncate
        # the answer to an empty string.
        max_tokens=1500,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    answer = (res.choices[0].message.content or "").strip()
    return RAGQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts),
    )
