"""
rag.py — RAG orchestrator for SAP Contract Intelligence Assistant.
Uses the new google.genai SDK for embeddings and generation.
"""

import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from google.genai import types

from config import (
    FAISS_INDEX_DIR,
    GENERATION_MODEL,
    SYSTEM_PROMPT,
    STYLE_INSTRUCTIONS,
    DEFAULT_TOP_K,
)
from ingest import embed_query, load_index as _load_index, get_client

# ── Index Loading ────────────────────────────────────────────────────────────

_cached_index = None
_cached_chunks = None


def load_index(force_reload: bool = False, index_dir: Optional[Path] = None):
    """Load (and cache) the FAISS index and chunk metadata."""
    global _cached_index, _cached_chunks
    if _cached_index is None or force_reload:
        _cached_index, _cached_chunks = _load_index(index_dir or FAISS_INDEX_DIR)
    return _cached_index, _cached_chunks


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(
    query: str, top_k: int = DEFAULT_TOP_K, index_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Embed query, search FAISS, return top-k chunks with scores."""
    index, chunks = load_index(index_dir=index_dir)
    if index is None or chunks is None:
        return []

    query_vec = embed_query(query)
    faiss.normalize_L2(query_vec)

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)
    return results


# ── Context Assembly ─────────────────────────────────────────────────────────

def _build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a structured context block."""
    if not chunks:
        return "(No relevant context was retrieved from the document.)"
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"--- Chunk {i} | Page {chunk['page']} | "
            f"Section: {chunk['section']} | Relevance: {chunk['score']:.2f} ---"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


# ── Answer Generation ────────────────────────────────────────────────────────

def generate_answer(
    query: str, chunks: List[Dict[str, Any]], style: str = "detailed",
) -> str:
    """Send retrieved context and user query to Gemini for generation."""
    context_block = _build_context_block(chunks)
    style_instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["detailed"])
    system_instruction = SYSTEM_PROMPT.format(
        style_instruction=style_instruction, context=context_block,
    )

    client = get_client()
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens=2048,
        )
    )
    return response.text


# ── Full RAG Pipeline ────────────────────────────────────────────────────────

def ask(
    query: str, top_k: int = DEFAULT_TOP_K, style: str = "detailed",
    index_dir: Optional[Path] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Full RAG pipeline: retrieve → generate. Returns (answer, chunks)."""
    chunks = retrieve(query, top_k=top_k, index_dir=index_dir)
    answer = generate_answer(query, chunks, style=style)
    return answer, chunks
