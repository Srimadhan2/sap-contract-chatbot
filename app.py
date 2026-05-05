"""
app.py — Streamlit frontend for SAP Contract Intelligence Assistant.

Premium enterprise-style RAG chatbot with dark glassmorphism UI,
ChatGPT-style conversation layout, and clause-level source citations.
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
from ingest import ingest_pdf, reset_client
from rag import ask, load_index as rag_load_index

# ── Environment & API Key Setup ─────────────────────────────────────────────
load_dotenv()


def _get_api_key() -> str:
    """Resolve the OpenAI API key from Streamlit secrets or environment."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY", "")


# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAP Contract Intelligence Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Load Custom CSS ──────────────────────────────────────────────────────────
def _load_css():
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    st.markdown(
        '<div class="floating-orb orb-1"></div>'
        '<div class="floating-orb orb-2"></div>',
        unsafe_allow_html=True,
    )


_load_css()

# ── Session State Initialization ─────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("index_ready", False),
    ("doc_name", None),
    ("chunk_count", 0),
    ("pending_prompt", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Configure API Key ────────────────────────────────────────────────────────
api_key = _get_api_key()
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    reset_client()

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # App branding
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <div style="font-size: 2.2rem; margin-bottom: 0.2rem;">⚖️</div>
            <div style="font-size: 1.1rem; font-weight: 700;
                        background: linear-gradient(135deg, #6366f1, #a855f7);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        background-clip: text;">
                SAP Contract Intelligence
            </div>
            <div style="font-size: 0.72rem; color: #9597b0; margin-top: 0.2rem;">
                AI-Powered Contract Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── API Key Input (if not set) ───────────────────────────────────────
    if not api_key:
        st.markdown("### 🔑 API Key Required")
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Enter your API key...",
            help="Get your key at https://platform.openai.com/api-keys",
        )
        if api_key_input:
            api_key = api_key_input
            os.environ["OPENAI_API_KEY"] = api_key
            reset_client()
        st.divider()

    # ── Document Management ──────────────────────────────────────────────
    st.markdown("### 📄 Document")
    uploaded_file = st.file_uploader(
        "Upload Contract PDF",
        type=["pdf"],
        help="Upload an SAP procurement contract or any legal PDF.",
        key="pdf_uploader",
    )

    if st.session_state.doc_name:
        st.markdown(
            f'<div class="status-badge success">✅ {st.session_state.doc_name}</div>'
            f'<div style="color:#9597b0;font-size:0.75rem;margin-top:0.3rem;">'
            f'{st.session_state.chunk_count} chunks indexed</div>',
            unsafe_allow_html=True,
        )

    # Process uploaded PDF
    if uploaded_file is not None:
        current_name = uploaded_file.name
        if current_name != st.session_state.doc_name:
            if not api_key:
                st.error("⚠️ Please enter your OpenAI API key first.")
            else:
                with st.spinner("📊 Ingesting document..."):
                    try:
                        pdf_bytes = io.BytesIO(uploaded_file.read())
                        chunks = ingest_pdf(
                            pdf_bytes,
                            chunk_size=st.session_state.get(
                                "chunk_size", DEFAULT_CHUNK_SIZE
                            ),
                            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                        )
                        rag_load_index(force_reload=True)
                        st.session_state.index_ready = True
                        st.session_state.doc_name = current_name
                        st.session_state.chunk_count = len(chunks)
                        st.success(f"✅ Indexed **{len(chunks)}** chunks!")
                    except Exception as e:
                        st.error(f"❌ Ingestion failed: {e}")

    # Rebuild index button
    if st.session_state.doc_name:
        if st.button("🔄 Rebuild Index", use_container_width=True):
            if uploaded_file is not None:
                with st.spinner("♻️ Rebuilding..."):
                    try:
                        uploaded_file.seek(0)
                        pdf_bytes = io.BytesIO(uploaded_file.read())
                        chunks = ingest_pdf(
                            pdf_bytes,
                            chunk_size=st.session_state.get(
                                "chunk_size", DEFAULT_CHUNK_SIZE
                            ),
                            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                        )
                        rag_load_index(force_reload=True)
                        st.session_state.chunk_count = len(chunks)
                        st.success(f"✅ Rebuilt with **{len(chunks)}** chunks!")
                    except Exception as e:
                        st.error(f"❌ Rebuild failed: {e}")
            else:
                st.warning("Please re-upload the PDF to rebuild.")

    st.divider()

    # ── Retrieval Settings ───────────────────────────────────────────────
    st.markdown("### ⚙️ Settings")
    chunk_size = st.slider(
        "Chunk Size (chars)", 400, 1500, DEFAULT_CHUNK_SIZE, 100,
        help="Larger chunks = more context; smaller = more precision.",
        key="chunk_size",
    )
    top_k = st.slider(
        "Top-K Results", 1, 10, DEFAULT_TOP_K,
        help="Number of relevant chunks to retrieve per question.",
        key="top_k",
    )
    answer_style = st.selectbox(
        "Answer Style",
        options=list(ANSWER_STYLES.keys()),
        index=0,
        help="Controls the verbosity and format of the AI's responses.",
        key="answer_style",
    )
    st.divider()

    # ── Sample Prompts ───────────────────────────────────────────────────
    st.markdown("### 💡 Sample Prompts")
    st.caption("Click any prompt to send it instantly.")
    for i, prompt in enumerate(SAMPLE_PROMPTS):
        st.markdown('<div class="sample-prompt-btn">', unsafe_allow_html=True)
        if st.button(prompt, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_prompt = prompt
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── Clear Chat ───────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<div style="text-align:center;padding-top:1rem;color:#5e607a;font-size:0.7rem;">'
        "Powered by OpenAI & FAISS<br/>Built with Streamlit</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT AREA
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown(
    """
    <div class="main-header">
        <h1>⚖️ SAP Contract Intelligence Assistant</h1>
        <div class="subtitle">
            Ask questions about your SAP procurement contracts —
            get instant, AI-powered answers with clause-level citations.
        </div>
        <div class="accent-line"></div>
        <div class="disclaimer">
            ⚠️ This tool is for document understanding only. It does not
            constitute legal advice. Always consult qualified legal counsel
            for actionable decisions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Status Metrics ───────────────────────────────────────────────────────
if st.session_state.index_ready:
    c1, c2, c3 = st.columns(3)
    c1.metric("📄 Document", st.session_state.doc_name or "None")
    c2.metric("🧩 Chunks Indexed", st.session_state.chunk_count)
    c3.metric("🔍 Retrieval Top-K", st.session_state.get("top_k", DEFAULT_TOP_K))

st.markdown("---")

# ── Chat History Display ─────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Retrieved Sources & Citations"):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-header">📌 Page {src["page"]} · '
                        f'{src["section"]} · Relevance: {src["score"]:.2f}</div>'
                        f'<div class="source-text">'
                        f'{src["text"][:400]}{"..." if len(src["text"]) > 400 else ""}'
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )


# ── Query Processing ─────────────────────────────────────────────────────
def _process_query(query: str):
    """Process a user query through the RAG pipeline and update chat."""
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if not api_key:
        err = "⚠️ Please enter your OpenAI API key in the sidebar."
        st.session_state.messages.append(
            {"role": "assistant", "content": err, "sources": []}
        )
        with st.chat_message("assistant"):
            st.markdown(err)
        return

    if not st.session_state.index_ready:
        err = "📄 Please upload a contract PDF first using the sidebar."
        st.session_state.messages.append(
            {"role": "assistant", "content": err, "sources": []}
        )
        with st.chat_message("assistant"):
            st.markdown(err)
        return

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching contract & generating answer..."):
            try:
                style = ANSWER_STYLES.get(
                    st.session_state.get("answer_style", "Detailed"), "detailed"
                )
                answer, chunks = ask(
                    query=query,
                    top_k=st.session_state.get("top_k", DEFAULT_TOP_K),
                    style=style,
                )

                st.markdown(answer, unsafe_allow_html=True)

                # Copyable answer block
                st.code(answer, language=None)

                if chunks:
                    with st.expander("📚 Retrieved Sources & Citations"):
                        for src in chunks:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<div class="source-header">📌 Page {src["page"]} · '
                                f'{src["section"]} · Relevance: {src["score"]:.2f}</div>'
                                f'<div class="source-text">'
                                f'{src["text"][:400]}{"..." if len(src["text"]) > 400 else ""}'
                                f"</div></div>",
                                unsafe_allow_html=True,
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
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                err = f"❌ An error occurred: {str(e)}"
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err, "sources": []}
                )


# ── Chat Input ───────────────────────────────────────────────────────────
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    _process_query(prompt)

if user_input := st.chat_input("Ask a question about the contract..."):
    _process_query(user_input)

# ── Empty State ──────────────────────────────────────────────────────────
if not st.session_state.messages and not st.session_state.index_ready:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding:3rem 2rem; margin-top:1rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">📑</div>
            <div style="font-size:1.2rem; font-weight:600; color:#e8e8f0; margin-bottom:0.5rem;">
                Upload a Contract to Get Started
            </div>
            <div style="color:#9597b0; font-size:0.92rem; max-width:500px; margin:0 auto;">
                Use the sidebar to upload an SAP procurement contract PDF.
                The AI will index the document and you can start asking questions
                about clauses, obligations, penalties, and more.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif not st.session_state.messages and st.session_state.index_ready:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding:3rem 2rem; margin-top:1rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">✅</div>
            <div style="font-size:1.2rem; font-weight:600; color:#e8e8f0; margin-bottom:0.5rem;">
                Document Indexed — Ready to Chat!
            </div>
            <div style="color:#9597b0; font-size:0.92rem; max-width:500px; margin:0 auto;">
                Your contract has been processed. Type a question below or
                click a sample prompt in the sidebar to begin analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
