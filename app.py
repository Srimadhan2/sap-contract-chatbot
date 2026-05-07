"""
app.py — Streamlit frontend for SAP Contract Intelligence Assistant
"""

import os
import io
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    ANSWER_STYLES,
    SAMPLE_PROMPTS,
)

from ingest import ingest_pdf
from rag import ask, load_index as rag_load_index

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

# =========================================================
# API KEY
# =========================================================

def _get_api_key() -> str:

    try:
        return st.secrets["GEMINI_API_KEY"]

    except Exception:
        pass

    return os.environ.get("GEMINI_API_KEY", "")


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SAP Contract Intelligence Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# LOAD CSS
# =========================================================

def _load_css():

    css_path = (
        Path(__file__).parent
        / "assets"
        / "style.css"
    )

    if css_path.exists():

        with open(css_path) as f:

            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True,
            )


_load_css()

# =========================================================
# SESSION STATE
# =========================================================

for key, default in [
    ("messages", []),
    ("index_ready", False),
    ("doc_name", None),
    ("chunk_count", 0),
    ("pending_prompt", None),
]:

    if key not in st.session_state:
        st.session_state[key] = default

# =========================================================
# API KEY SETUP
# =========================================================

api_key = _get_api_key()

if api_key:
    os.environ["GEMINI_API_KEY"] = api_key

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ⚖️ SAP Contract Intelligence")

    st.divider()

    # =====================================================
    # API KEY
    # =====================================================

    st.markdown("### 🔑 Gemini API Key")

    api_key_input = st.text_input(
        "Enter API Key",
        value=api_key if api_key else "",
        type="password",
    )

    if api_key_input:

        api_key = api_key_input

        os.environ["GEMINI_API_KEY"] = api_key

    st.divider()

    # =====================================================
    # FILE UPLOAD
    # =====================================================

    st.markdown("### 📄 Upload Contract")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="pdf_uploader",
    )

    # =====================================================
    # PDF PROCESSING
    # =====================================================

    if uploaded_file is not None:

        current_name = uploaded_file.name

        # Prevent repeated ingestion
        if (
            current_name != st.session_state.doc_name
        ):

            if not api_key:

                st.error(
                    "⚠️ Please enter Gemini API key."
                )

            else:

                with st.spinner(
                    "🔄 Processing PDF..."
                ):

                    try:

                        uploaded_file.seek(0)

                        pdf_bytes = io.BytesIO(
                            uploaded_file.read()
                        )

                        chunks = ingest_pdf(
                            pdf_bytes,
                            chunk_size=st.session_state.get(
                                "chunk_size",
                                DEFAULT_CHUNK_SIZE,
                            ),
                            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                        )

                        rag_load_index(
                            force_reload=True
                        )

                        st.session_state.index_ready = True

                        st.session_state.doc_name = current_name

                        st.session_state.chunk_count = len(
                            chunks
                        )

                        st.success(
                            f"✅ Indexed "
                            f"{len(chunks)} chunks!"
                        )

                    except Exception as e:

                        st.error(
                            f"❌ Ingestion failed: {e}"
                        )

    # =====================================================
    # DOCUMENT STATUS
    # =====================================================

    if st.session_state.doc_name:

        st.success(
            f"📄 {st.session_state.doc_name}"
        )

        st.caption(
            f"{st.session_state.chunk_count} chunks indexed"
        )

    st.divider()

    # =====================================================
    # SETTINGS
    # =====================================================

    st.markdown("### ⚙️ Settings")

    chunk_size = st.slider(
        "Chunk Size",
        400,
        1500,
        DEFAULT_CHUNK_SIZE,
        100,
        key="chunk_size",
    )

    top_k = st.slider(
        "Top K Retrieval",
        1,
        10,
        DEFAULT_TOP_K,
        key="top_k",
    )

    answer_style = st.selectbox(
        "Answer Style",
        options=list(ANSWER_STYLES.keys()),
        index=0,
        key="answer_style",
    )

    st.divider()

    # =====================================================
    # SAMPLE PROMPTS
    # =====================================================

    st.markdown("### 💡 Sample Prompts")

    for i, prompt in enumerate(
        SAMPLE_PROMPTS
    ):

        if st.button(
            prompt,
            key=f"sample_{i}",
            use_container_width=True,
        ):

            st.session_state.pending_prompt = prompt

    st.divider()

    # =====================================================
    # CLEAR CHAT
    # =====================================================

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

# =========================================================
# MAIN HEADER
# =========================================================

st.title("⚖️ SAP Contract Intelligence Assistant")

st.caption(
    "AI-powered enterprise contract analysis using "
    "Gemini + FAISS RAG pipeline."
)

st.markdown("---")

# =========================================================
# METRICS
# =========================================================

if st.session_state.index_ready:

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "📄 Document",
        st.session_state.doc_name,
    )

    c2.metric(
        "🧩 Chunks",
        st.session_state.chunk_count,
    )

    c3.metric(
        "🔍 Top-K",
        st.session_state.get(
            "top_k",
            DEFAULT_TOP_K,
        ),
    )

# =========================================================
# CHAT HISTORY
# =========================================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if (
            msg["role"] == "assistant"
            and msg.get("sources")
        ):

            with st.expander(
                "📚 Retrieved Sources"
            ):

                for src in msg["sources"]:

                    st.markdown(
                        f"""
### 📌 Page {src["page"]}

**Section:** {src["section"]}

**Relevance:** {src["score"]:.2f}

{src["text"][:500]}
"""
                    )

# =========================================================
# QUERY PROCESSING
# =========================================================

def _process_query(query: str):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    # =====================================================
    # VALIDATIONS
    # =====================================================

    if not api_key:

        err = (
            "⚠️ Please enter Gemini API key."
        )

        st.error(err)

        return

    if not st.session_state.index_ready:

        err = (
            "📄 Please upload PDF first."
        )

        st.error(err)

        return

    # =====================================================
    # RAG PIPELINE
    # =====================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "🔍 Retrieving answer..."
        ):

            try:

                style = ANSWER_STYLES.get(
                    st.session_state.get(
                        "answer_style",
                        "Detailed",
                    ),
                    "detailed",
                )

                answer, chunks = ask(
                    query=query,
                    top_k=st.session_state.get(
                        "top_k",
                        DEFAULT_TOP_K,
                    ),
                    style=style,
                )

                st.markdown(answer)

                # =========================================
                # SOURCES
                # =========================================

                if chunks:

                    with st.expander(
                        "📚 Retrieved Sources"
                    ):

                        for src in chunks:

                            st.markdown(
                                f"""
### 📌 Page {src["page"]}

**Section:** {src["section"]}

**Relevance:** {src["score"]:.2f}

{src["text"][:500]}
"""
                            )

                sources = [
                    {
                        "page": c["page"],
                        "section": c["section"],
                        "score": c["score"],
                        "text": c["text"],
                    }
                    for c in chunks
                ]

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as e:

                st.error(
                    f"❌ Error: {str(e)}"
                )

# =========================================================
# SAMPLE PROMPT AUTO RUN
# =========================================================

if st.session_state.pending_prompt:

    prompt = st.session_state.pending_prompt

    st.session_state.pending_prompt = None

    _process_query(prompt)

# =========================================================
# CHAT INPUT
# =========================================================

if user_input := st.chat_input(
    "Ask questions about the contract..."
):

    _process_query(user_input)

# =========================================================
# EMPTY STATE
# =========================================================

if (
    not st.session_state.messages
    and not st.session_state.index_ready
):

    st.info(
        "📄 Upload a PDF contract to begin."
    )

elif (
    not st.session_state.messages
    and st.session_state.index_ready
):

    st.success(
        "✅ Document indexed successfully. "
        "Ask questions below."
    )