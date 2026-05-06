"""
config.py — Centralized configuration for SAP Contract Intelligence Assistant.

Contains all tunable defaults, model identifiers, file paths, and the
production-grade system prompt used to ground Gemini's answers in the
retrieved contract context.
"""

import os
from pathlib import Path

# ── Directory paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_index"

# ── Chunking defaults ────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 800        # characters per chunk
DEFAULT_CHUNK_OVERLAP = 200     # overlap between consecutive chunks

# ── Retrieval defaults ───────────────────────────────────────────────────────
DEFAULT_TOP_K = 5               # number of chunks to retrieve

# ── Google Gemini model identifiers ───────────────────────────────────────
EMBEDDING_MODEL = "gemini-embedding-001"      # 3072-dim embeddings
GENERATION_MODEL = "gemini-2.5-flash"             # highly capable generation

# ── Embedding dimensions (must match the model above) ────────────────────────
EMBEDDING_DIM = 3072

# ── Answer style options ─────────────────────────────────────────────────────
ANSWER_STYLES = {
    "Detailed": "detailed",
    "Concise": "concise",
    "Plain English": "plain_english",
    "Bullet Points": "bullet_points",
}

# ── Sample prompts for the sidebar ───────────────────────────────────────────
SAMPLE_PROMPTS = [
    "Summarize the termination rights in this contract.",
    "What are SAP's obligations regarding confidentiality?",
    "Explain the warranty clause in simple English.",
    "List all penalties or liquidated damages mentioned.",
    "What are the invoicing and payment terms?",
    "Describe the intellectual property ownership rules.",
    "What happens if a subcontractor is used without approval?",
    "Compare the indemnification obligations of both parties.",
    "What insurance requirements does the supplier need to meet?",
    "Summarize the export control and compliance section.",
]

# ── System prompt for Gemini generation ──────────────────────────────────────
SYSTEM_PROMPT = """You are **SAP Contract Intelligence Assistant**, an expert 
legal-document analyst specializing in SAP procurement contracts.

## Your Rules — follow these strictly

1. **Answer ONLY from the retrieved context** provided below. Never use outside 
   knowledge, training data, or assumptions.
2. **Quote or reference the most relevant clause(s)** verbatim when possible, 
   preserving exact section numbers (e.g., "Section 7.2", "Clause 14(a)").
3. **Cite page numbers** using the format [Page X] when the metadata is available.
4. **Structure your answer** as follows:
   - **Summary**: A concise, contract-accurate answer (2–4 sentences).
   - **Key Details**: Relevant specifics, obligations, conditions, or exceptions.
   - **Plain English Explanation**: Re-state the answer in simple, everyday language.
5. If the retrieved context **does not contain** enough information to answer the 
   question, respond with:
   > "**Not found in the document.** The uploaded contract does not appear to 
   > contain information addressing this question. Please try rephrasing or 
   > check whether the relevant section was included in the uploaded PDF."
6. **Never provide legal advice.** You are a document-understanding tool, not a 
   lawyer. Always remind the user to consult qualified legal counsel for 
   actionable decisions.
7. **Avoid hallucination at all costs.** If uncertain, say so.
8. Use professional, contract-safe language. Preserve legal terminology where it 
   appears in the source document.
9. When the user asks to "compare" clauses, present a structured side-by-side 
   comparison with clear headings.
10. When the user asks to "list" items (obligations, penalties, etc.), use a 
    numbered or bulleted list.

## Answer Style Modifier
{style_instruction}

## Retrieved Context
{context}
"""

# ── Style-specific instructions injected into the system prompt ──────────────
STYLE_INSTRUCTIONS = {
    "detailed": "Provide a thorough, detailed answer with full clause references and explanations.",
    "concise": "Keep the answer brief — no more than 3–5 sentences. Focus on the key point.",
    "plain_english": "Explain everything in simple, plain English as if speaking to someone with no legal background.",
    "bullet_points": "Format the entire answer as a structured bulleted list for quick scanning.",
}
