# =========================================================
# ingest.py — FINAL SAP BTP SAFE VERSION
# =========================================================

import os
import re
import time
import pickle
from pathlib import Path
from typing import Optional

# =========================================================
# SAP BTP NETWORK FIX
# =========================================================

os.environ["GRPC_DNS_RESOLVER"] = "native"
os.environ["GLOG_minloglevel"] = "2"

# =========================================================
# LOAD ENV
# =========================================================

from dotenv import load_dotenv
load_dotenv()

# =========================================================
# IMPORTS
# =========================================================

import numpy as np
import faiss

from google import genai
from PyPDF2 import PdfReader

# =========================================================
# CONFIG
# =========================================================

FAISS_INDEX_DIR = Path(
    "faiss_index"
)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

EMBEDDING_MODEL = (
    "gemini-embedding-001"
)

# =========================================================
# GEMINI CLIENT
# =========================================================

_client: Optional[
    genai.Client
] = None


def reset_client():

    global _client

    _client = None


def get_client():

    global _client

    if _client is None:

        api_key = os.environ.get(
            "GEMINI_API_KEY",
            ""
        )

        if not api_key:

            raise ValueError(
                "❌ GEMINI_API_KEY not found"
            )

        _client = genai.Client(
            api_key=api_key,
            http_options={
                "api_version": "v1beta"
            }
        )

    return _client


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_text_from_pdf(
    pdf_path_or_file
):

    reader = PdfReader(
        pdf_path_or_file
    )

    pages = []

    for i, page in enumerate(
        reader.pages
    ):

        text = (
            page.extract_text() or ""
        )

        if text.strip():

            pages.append({
                "page": i + 1,
                "text": text
            })

    return pages


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# CHUNKING
# =========================================================

def chunk_pages(
    pages,
    chunk_size=1000,
    chunk_overlap=200
):

    chunks = []

    chunk_id = 0

    for page_info in pages:

        text = clean_text(
            page_info["text"]
        )

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[
                start:end
            ]

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page": page_info["page"],
                "section": "General",
            })

            chunk_id += 1

            start += (
                chunk_size
                - chunk_overlap
            )

    return chunks


# =========================================================
# EMBEDDINGS
# =========================================================

def embed_texts(
    texts,
    batch_size=10
):

    client = get_client()

    all_embeddings = []

    for i in range(
        0,
        len(texts),
        batch_size
    ):

        batch = texts[
            i:i + batch_size
        ]

        print(
            f"🧠 Embedding batch "
            f"{i // batch_size + 1}"
        )

        result = (
            client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
            )
        )

        for embedding in (
            result.embeddings
        ):

            all_embeddings.append(
                embedding.values
            )

        time.sleep(0.5)

    return np.array(
        all_embeddings,
        dtype=np.float32
    )


def embed_query(text):

    client = get_client()

    result = (
        client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )
    )

    vector = np.array(
        result.embeddings[0].values,
        dtype=np.float32,
    ).reshape(1, -1)

    return vector


# =========================================================
# FAISS
# =========================================================

def build_faiss_index(
    embeddings
):

    faiss.normalize_L2(
        embeddings
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# =========================================================
# SAVE INDEX
# =========================================================

def save_index(
    index,
    chunks
):

    FAISS_INDEX_DIR.mkdir(
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(
            FAISS_INDEX_DIR
            / "index.faiss"
        )
    )

    with open(
        FAISS_INDEX_DIR
        / "chunks.pkl",
        "wb"
    ) as f:

        pickle.dump(
            chunks,
            f
        )


# =========================================================
# LOAD INDEX
# =========================================================

def load_index(force_reload=False):

    index_path = (
        FAISS_INDEX_DIR
        / "index.faiss"
    )

    chunks_path = (
        FAISS_INDEX_DIR
        / "chunks.pkl"
    )

    if (
        not index_path.exists()
        or not chunks_path.exists()
    ):

        return None, None

    index = faiss.read_index(
        str(index_path)
    )

    with open(
        chunks_path,
        "rb"
    ) as f:

        chunks = pickle.load(f)

    return index, chunks


# =========================================================
# MAIN INGESTION
# =========================================================

def ingest_pdf(
    pdf_path_or_file,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
):

    print(
        "\n🚀 STARTING PDF INGESTION"
    )

    # =====================================
    # EXTRACT TEXT
    # =====================================

    print(
        "\n📄 Extracting PDF text..."
    )

    pages = extract_text_from_pdf(
        pdf_path_or_file
    )

    print(
        f"✅ Pages extracted: "
        f"{len(pages)}"
    )

    # =====================================
    # CHUNKING
    # =====================================

    print(
        "\n✂ Creating chunks..."
    )

    chunks = chunk_pages(
    pages,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
    print(
        f"✅ Chunks created: "
        f"{len(chunks)}"
    )

    # =====================================
    # EMBEDDINGS
    # =====================================

    print(
        "\n🧠 Generating embeddings..."
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embed_texts(
        texts
    )

    print(
        f"✅ Embeddings shape: "
        f"{embeddings.shape}"
    )

    # =====================================
    # BUILD INDEX
    # =====================================

    print(
        "\n📦 Building FAISS index..."
    )

    index = build_faiss_index(
        embeddings
    )

    print(
        f"✅ Total vectors stored: "
        f"{index.ntotal}"
    )

    # =====================================
    # SAVE
    # =====================================

    print(
        "\n💾 Saving vector DB..."
    )

    save_index(
        index,
        chunks
    )

    print(
        "\n🎉 INGESTION COMPLETED"
    )

    return chunks