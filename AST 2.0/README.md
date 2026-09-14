# Analyst Assistant ⚡

An AI-powered data intelligence system for natural language CSV analysis and document Q&A. Built with a modular Python backend, a hardened code execution sandbox, and a Retrieval-Augmented Generation (RAG) pipeline — deployed as a public web app via Streamlit Cloud and Groq API.

---

## Live Demo

> Add your Streamlit Cloud URL here once deployed.

---

## What It Does

**CSV Analysis Tab**
- Upload any CSV and ask questions in plain English
- Auto-cleans the dataset before analysis (two-stage pipeline)
- Generates and executes pandas/Plotly code in a sandboxed environment
- Produces interactive charts, tables, and scalar results
- Generates ranked business insights and deep-drill explanations from the data

**Document Intelligence Tab**
- Upload PDF, DOCX, TXT, or Markdown files
- Chunks and embeds the document locally using `all-MiniLM-L6-v2`
- Answers questions using RAG — retrieves relevant passages, sends only those to the LLM
- Extracts and ranks key insights from the document automatically

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        app.py (main2.py)                    │
│                     Streamlit UI layer                      │
└──────────┬──────────────────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────────────────┐
    │                   Query Router                       │
    │  detect_shortcut() → direct pandas (no LLM call)   │
    │  else → build_analysis_prompt() → LLM               │
    └──────┬───────────────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────────────────┐
    │              llm_provider.py                         │
    │  Single source of truth for all LLM calls           │
    │  ├── llm_code_call()    → llama-3.1-8b-instant      │
    │  ├── llm_chat_call()    → llama-3.3-70b-versatile   │
    │  └── llm_uncached_code_call() → retries/cleaning    │
    │  Provider: Groq API (free tier)                      │
    └──────┬───────────────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────────────────┐
    │              sandbox.py                              │
    │  Two-layer code execution guard                      │
    │  ├── Layer 1: AST analysis (imports, dunders,        │
    │  │            blocked builtins, module access)       │
    │  └── Layer 2: String pattern blocklist               │
    │  exec() runs with __builtins__: {} (empty namespace) │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │              rag_engine.py                           │
    │  Document parsing → chunking → embedding → retrieval│
    │  Embedding model: all-MiniLM-L6-v2 (CPU, local)    │
    │  Vector store: ChromaDB (in-memory)                  │
    │  LLM answers via llm_provider (Groq API)            │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │              cleaning.py                             │
    │  Stage 1: Deterministic rule-based pipeline         │
    │  ├── Header promotion, column normalisation         │
    │  ├── Currency, percentage, thousands separators     │
    │  ├── Sentinel null detection and replacement        │
    │  ├── Duplicate removal, datetime parsing            │
    │  └── Null filling: median (numeric), Unknown (cat.) │
    │  Stage 2: LLM pass for remaining anomalies          │
    └─────────────────────────────────────────────────────┘
```

---

## Module Reference

| File | Role |
|---|---|
| `app.py` | Streamlit UI — CSV and Document tabs, sidebar, session state |
| `llm_provider.py` | Centralised Groq API client — all LLM calls route through here |
| `query_engine.py` | Shortcut layer, prompt builder, code post-processing |
| `cleaning.py` | Deterministic rule-based data cleaning pipeline |
| `profiler.py` | Dataset statistical profiler, insight/explain prompt builders |
| `sandbox.py` | AST + pattern-based code safety guard, hardened `exec()` |
| `rag_engine.py` | Document parsing, chunking, embedding, ChromaDB vector store, RAG Q&A |
| `ui.py` | Reusable Streamlit rendering helpers — insight boxes, charts, confidence scoring |
| `eval.py` | Offline evaluation harness for deterministic components |

---

## Key Design Decisions

**Dual-model routing**
Code generation (pandas, analysis, retries) goes to `llama-3.1-8b-instant` — 560 tokens/sec, deterministic at temperature 0. Business insights and RAG answers go to `llama-3.3-70b-versatile` — better reasoning quality where speed matters less.

**Shortcut layer before LLM**
Common queries (`how many rows`, `show columns`, `describe`, `sort by`, `correlation`, etc.) are detected and answered with direct pandas code — no LLM call, no latency, deterministic output.

**Two-stage cleaning**
Rule-based pass first (no LLM, no latency, handles all pattern-matchable issues), then a single targeted LLM pass only for what the rules missed. The LLM never rewrites work already done.

**Sandboxed execution**
Generated code passes through AST analysis + string pattern matching before `exec()`. The execution namespace has `__builtins__: {}` — no file I/O, no imports, no subprocess, no eval. Blocked at two independent layers.

**RAG stays local**
Document embeddings run on CPU via `all-MiniLM-L6-v2` (~90MB model). Only retrieved passage text goes to the API — raw document content never leaves the server. ChromaDB runs in-memory with no external server.

**Single provider abstraction**
All LLM calls go through `llm_provider.py`. Switching inference providers (Groq → Anthropic → OpenAI) requires changing two constants in one file.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM Inference | Groq API (free tier) |
| Code Model | `llama-3.1-8b-instant` |
| Chat / Reasoning Model | `llama-3.3-70b-versatile` |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (in-memory) |
| Data Processing | pandas, numpy |
| Visualisation | Plotly (interactive), matplotlib (fallback) |
| Document Parsing | pypdf, python-docx |
| Language | Python 3.11+ |
| Deployment | Streamlit Cloud |

---

## Local Development Setup

**Prerequisites:** Python 3.11+, a free Groq API key from [console.groq.com](https://console.groq.com)

```bash
# 1. Clone the repo
git clone https://github.com/your-username/analyst-assistant.git
cd analyst-assistant

# 2. Install CPU-only torch first (avoids torchaudio conflicts on Windows)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install all other dependencies
pip install -r requirements.txt

# 4. Create your .env file
cp .env.example .env
# Open .env and add your key: GROQ_API_KEY="gsk_..."

# 5. Run
streamlit run app.py
```

---

## Deployment (Streamlit Cloud)

1. Push this repo to GitHub (public repository)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account
3. Click **Create app** → select this repo → set main file to `app.py`
4. Click **Advanced settings** → **Secrets** → paste:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```
5. Click **Deploy** — live in ~3 minutes

No server setup, no Docker, no infrastructure. The embedding model downloads automatically on first run (~90MB, cached after that).

---

## Running the Eval Harness

Tests all deterministic components without any LLM calls or API keys needed:

```bash
python eval.py                          # all test suites
python eval.py --suite shortcuts        # shortcut detection only
python eval.py --suite cleaning         # data cleaning pipeline only
python eval.py --suite security         # sandbox security only
python eval.py --csv your_data.csv      # test against your own dataset
python eval.py --verbose                # show generated code for each test
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key — get free at console.groq.com |

For local development, set this in `.env`. For Streamlit Cloud, set it in App Settings → Secrets.

---

## Project Structure

```
analyst-assistant/
├── app.py                 # Main Streamlit application (renamed from main2.py)
├── llm_provider.py        # Centralised LLM call layer (Groq API)
├── query_engine.py        # Query routing, prompts, code cleanup
├── cleaning.py            # Deterministic data cleaning pipeline
├── profiler.py            # Dataset profiler and insight prompt builders
├── sandbox.py             # Code safety guard and hardened exec
├── rag_engine.py          # RAG pipeline — parse, chunk, embed, retrieve, answer
├── ui.py                  # Streamlit rendering helpers
├── eval.py                # Offline evaluation harness
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── .gitignore             # Excludes .env, __pycache__, chromadb data
└── .streamlit/
    └── config.toml        # Theme and server configuration
```

---

## Security Notes

- Generated code is validated by AST analysis and string pattern matching before execution
- `exec()` runs with an empty builtins namespace — no access to file system, network, or system calls
- CSV data rows are never sent to the LLM — only schema and statistical summaries
- Document content stays on the server — only retrieved RAG passages go to the API
- API key is read from environment variables only, never hardcoded

---

## License

MIT
