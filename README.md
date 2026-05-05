# SAP Contract Intelligence Assistant

> **AI-powered contract analysis** — Upload SAP procurement PDFs and get instant, grounded answers with clause-level citations.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Google%20Gemini-2.0-4285F4?logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-009688)

---

## ✨ Features

- **RAG-powered Q&A** — Retrieval-Augmented Generation ensures every answer is grounded in the actual contract text
- **Clause-level citations** — Every answer includes page numbers, section references, and relevance scores
- **Multiple answer styles** — Detailed, Concise, Plain English, or Bullet Points
- **Smart chunking** — Section-aware text splitting preserves clause integrity
- **Premium UI** — Dark glassmorphism theme with smooth animations
- **Session persistence** — Chat history maintained across interactions
- **Configurable retrieval** — Adjust chunk size, top-k results, and answer style via the sidebar

---

## 🏗️ Architecture

```
PDF Upload → PyPDF2 Extraction → Text Cleaning → Section-Aware Chunking
    → Gemini Embeddings (text-embedding-004) → FAISS Index
    
User Question → Query Embedding → FAISS Top-K Retrieval
    → Context Assembly → Gemini 2.0 Flash Generation → Grounded Answer
```

### File Structure

```
sap-contract-chatbot/
├── app.py              # Streamlit frontend — premium UI, chat, sidebar
├── rag.py              # RAG orchestrator — retrieval + generation
├── ingest.py           # PDF ingestion — extract, chunk, embed, index
├── config.py           # Constants, prompts, and defaults
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
├── README.md           # This file
└── assets/
    └── style.css       # Premium dark theme CSS
```

---

## 🚀 Quick Start (Local)

### Prerequisites

- Python 3.9 or higher
- A [Google Gemini API key](https://aistudio.google.com/app/apikey) (free tier works)

### Step 1 — Clone & Install

```bash
git clone <your-repo-url>
cd sap-contract-chatbot

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Step 2 — Configure API Key

```bash
# Option A: Copy the template and fill in your key
cp .env.example .env
# Edit .env and replace 'your-gemini-api-key-here' with your actual key

# Option B: Set as environment variable
export GOOGLE_API_KEY="your-key-here"       # Linux/Mac
# $env:GOOGLE_API_KEY="your-key-here"       # PowerShell
```

### Step 3 — Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Upload a PDF and start asking questions!

---

## ☁️ Deploy to Streamlit Community Cloud

1. **Push to GitHub** — Push this repo to a public or private GitHub repository.

2. **Connect on Streamlit Cloud** — Go to [share.streamlit.io](https://share.streamlit.io), click "New app", and select your repo.

3. **Configure Secrets** — In the Streamlit Cloud dashboard, go to **Settings → Secrets** and add:

   ```toml
   GOOGLE_API_KEY = "your-gemini-api-key-here"
   ```

4. **Deploy** — Click "Deploy" and your app will be live in minutes.

> **Note:** The FAISS index is built at runtime when the user uploads a PDF, so no pre-built index needs to be committed to the repo.

---

## 💬 Sample Prompts

Try these questions after uploading an SAP procurement contract PDF:

| Category | Sample Question |
|----------|----------------|
| **Summarize** | "Summarize the termination rights in this contract." |
| **Explain** | "Explain the warranty clause in simple English." |
| **List** | "List all penalties or liquidated damages mentioned." |
| **Obligations** | "What are SAP's obligations regarding confidentiality?" |
| **Compare** | "Compare the indemnification obligations of both parties." |
| **Identify** | "What insurance requirements does the supplier need to meet?" |
| **Direct** | "What are the invoicing and payment terms?" |
| **Compliance** | "Summarize the export control and compliance section." |
| **Subcontracting** | "What happens if a subcontractor is used without approval?" |
| **IP** | "Describe the intellectual property ownership rules." |

---

## ⚙️ Configuration

All defaults can be adjusted in `config.py` or via the sidebar at runtime:

| Setting | Default | Description |
|---------|---------|-------------|
| Chunk Size | 800 chars | Maximum characters per text chunk |
| Chunk Overlap | 200 chars | Overlap between consecutive chunks |
| Top-K | 5 | Number of chunks retrieved per query |
| Embedding Model | `text-embedding-004` | Google Gemini embedding model |
| Generation Model | `gemini-2.0-flash` | Google Gemini generation model |
| Answer Style | Detailed | Response verbosity and format |

---

## 🔒 Disclaimer

This tool is designed for **document understanding and analysis only**. It does not provide legal advice. Always consult qualified legal counsel before making decisions based on contract interpretations.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
