"""
ingest.py — PDF ingestion pipeline for SAP Contract Intelligence Assistant.

Responsibilities:
  1. Extract text from PDF page-by-page (preserving page numbers).
  2. Clean and normalize the extracted text.
  3. Detect section headings and attach metadata.
  4. Chunk text intelligently with configurable size and overlap.
  5. Generate embeddings via Google Gemini embedding API (google.genai SDK).
  6. Build and persist a FAISS vector index alongside chunk metadata.
"""

import os
import re
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()


import numpy as np
import faiss
import openai

from PyPDF2 import PdfReader

from config import (
    FAISS_INDEX_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)


_client: Optional[openai.OpenAI] = None


def get_client() -> openai.OpenAI:
    """Return a cached OpenAI client, creating one if needed."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        _client = openai.OpenAI(api_key=api_key)
    return _client


def reset_client():
    """Force re-creation of the OpenAI client (e.g. after key change)."""
    global _client
    _client = None


# ── PDF Text Extraction ─────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path_or_file) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF file, returning a list of dicts:
        [{ "page": 1, "text": "..." }, ...]

    Accepts either a file path (str/Path) or a file-like object (from st.file_uploader).
    """
    reader = PdfReader(pdf_path_or_file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


# ── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize whitespace, fix hyphenation at line breaks, and remove
    common header/footer artifacts.
    """
    # Fix hyphenated words split across lines
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Replace multiple newlines with a single newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove common page-number patterns like "Page 1 of 20"
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    return text.strip()


# ── Section Detection ────────────────────────────────────────────────────────

SECTION_PATTERN = re.compile(
    r"^"
    r"(?:"
    r"(?:SECTION|ARTICLE|CLAUSE)\s+\d+[\.\:]?\s*"
    r"|"
    r"\d{1,2}\.\d{0,2}\.?\s+"
    r")"
    r"[A-Z]",
    re.MULTILINE,
)


def detect_section_title(text: str) -> str:
    """
    Attempt to extract the first section heading from a chunk of text.
    Returns the heading string or 'General' if none found.
    """
    match = SECTION_PATTERN.search(text)
    if match:
        line_start = match.start()
        line_end = text.find("\n", line_start)
        if line_end == -1:
            line_end = min(line_start + 120, len(text))
        heading = text[line_start:line_end].strip()
        return heading[:120] if len(heading) > 120 else heading
    return "General"


# ── Intelligent Chunking ─────────────────────────────────────────────────────

def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Split page texts into overlapping chunks, each carrying metadata:
        { "chunk_id": int, "text": str, "page": int, "section": str }
    """
    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for page_info in pages:
        page_num = page_info["page"]
        text = clean_text(page_info["text"])
        if not text:
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                boundary = _find_sentence_boundary(text, start, end)
                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()
            if chunk_text:
                section = detect_section_title(chunk_text)
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "page": page_num,
                    "section": section,
                })
                chunk_id += 1

            start = max(start + 1, end - chunk_overlap)

    return chunks


def _find_sentence_boundary(text: str, start: int, end: int) -> int:
    """
    Look backward from `end` for the last sentence-ending punctuation
    followed by whitespace. Returns the position just after the punctuation,
    or `start` if no boundary found.
    """
    search_region = text[start:end]
    for i in range(len(search_region) - 1, max(len(search_region) // 2, 0), -1):
        if search_region[i] in ".?!" and (
            i + 1 < len(search_region) and search_region[i + 1] in " \n\t"
        ):
            return start + i + 1
    return start


# ── Embedding Generation (openai SDK) ──────────────────────────────────

def embed_texts(texts: List[str], batch_size: int = 20) -> np.ndarray:
    """
    Generate embeddings for a list of texts using the OpenAI embedding API.
    Processes in batches to respect API rate limits.
    Returns a numpy array of shape (len(texts), EMBEDDING_DIM).
    """
    client = get_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        for data in result.data:
            all_embeddings.append(data.embedding)

        # Small delay to avoid rate-limiting on free tier
        if i + batch_size < len(texts):
            time.sleep(0.5)

    return np.array(all_embeddings, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """
    Generate an embedding for a single query string.
    """
    client = get_client()
    result = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    return np.array(result.data[0].embedding, dtype=np.float32).reshape(1, -1)


# ── FAISS Index Management ───────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS inner-product index from normalized embeddings.
    Inner product on L2-normalized vectors == cosine similarity.
    """
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    return index


def save_index(
    index: faiss.IndexFlatIP,
    chunks: List[Dict[str, Any]],
    index_dir: Optional[Path] = None,
):
    """Persist the FAISS index and chunk metadata to disk."""
    index_dir = index_dir or FAISS_INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "index.faiss"))
    with open(index_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_index(index_dir: Optional[Path] = None):
    """
    Load a previously saved FAISS index and chunk metadata.
    Returns (index, chunks) or (None, None) if files don't exist.
    """
    index_dir = index_dir or FAISS_INDEX_DIR
    index_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.pkl"

    if not index_path.exists() or not chunks_path.exists():
        return None, None

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ── Main Ingestion Pipeline ──────────────────────────────────────────────────

def ingest_pdf(
    pdf_path_or_file,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    index_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    End-to-end ingestion: PDF → text → chunks → embeddings → FAISS index.

    Args:
        pdf_path_or_file: Path to PDF or a file-like object.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        index_dir: Directory to save the FAISS index (default: FAISS_INDEX_DIR).

    Returns:
        List of chunk metadata dicts.
    """
    # Step 1: Extract text
    pages = extract_text_from_pdf(pdf_path_or_file)
    if not pages:
        raise ValueError(
            "No text could be extracted from the PDF. It may be scanned/image-only."
        )

    # Step 2: Chunk with metadata
    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError(
            "Chunking produced no results. The document may be empty after cleaning."
        )

    # Step 3: Generate embeddings
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # Step 4: Build and save FAISS index
    index = build_faiss_index(embeddings)
    save_index(index, chunks, index_dir=index_dir)

    return chunks
