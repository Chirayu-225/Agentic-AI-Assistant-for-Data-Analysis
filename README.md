# Agentic AI Assistant for Data Analysis

An agentic system for natural-language analysis of structured and unstructured data — upload a CSV or a document, ask questions in plain English, and get back real computed answers (tables, charts, scalars) or retrieval-grounded answers, not LLM guesses.

The repo has two parts:

| Folder | What it is |
|---|---|
| [`AST 2.0/`](./AST%202.0) | Python backend — the actual analysis, cleaning, RAG, and sandboxing logic. Runs standalone as a Streamlit app, or as a FastAPI service behind the Next.js frontend. |
| [`AST NextJS/`](./AST%20NextJS) | Next.js frontend — web UI that talks to the FastAPI service over a small JSON API. Currently tested against `localhost:8000`; not yet deployed. |

They're developed together but are independently runnable. You can use the backend on its own via Streamlit (`streamlit run app.py`), or run both together for the full web app.

---

## What it does

**CSV / spreadsheet analysis**
- Upload a CSV or Excel file and ask questions in plain English
- Two-stage cleaning: a deterministic rule-based pass, then an optional LLM agent pass for defects the rules can't catch (inconsistent category spellings, implausible values, etc.)
- Generates and runs pandas/Plotly code in a hardened sandbox, returning tables, scalars, or interactive charts
- One-click "auto-analyze": ranked business insights + a handful of charts chosen to support them, with follow-up Q&A against the same context
- Multi-table queries across several uploaded datasets at once, joined via real SQL (DuckDB) rather than an LLM trying to reason about multiple frames at once

**Document intelligence (RAG)**
- Upload a PDF, DOCX, TXT, or Markdown file
- Parses, chunks, and embeds it locally (`all-MiniLM-L6-v2`, CPU, no network call for the embedding step)
- Answers questions by retrieving relevant passages and sending only those to the LLM — raw document text never leaves the server except as cited context
- Auto-extracts ranked insights from the document
- Tables embedded inside PDFs are detected and extracted into real DataFrames, so questions about numbers in a table go through the same sandboxed computation as a CSV upload, instead of being eyeballed out of retrieved text

**Dataset profiling**
- Schema manifest, per-column stats, and a before/after cleaning report shown as an explicit, auditable step rather than a silent one

---

## Architecture

```
┌────────────────────────┐        ┌──────────────────────────────────────┐
│   AST NextJS (web UI)   │  HTTP  │        AST 2.0 (Python backend)       │
│                          │ ─────▶ │                                        │
│  CSV / doc upload        │        │  api.py — thin FastAPI layer:         │
│  (direct-to-Blob, so     │        │    secret-header auth, fetch file     │
│  large files skip the    │        │    from Blob, delegate to a service   │
│  4.5MB serverless limit) │        │    module, return JSON                │
│  Query panel, results,   │        │                                        │
│  charts, RAG panel       │        │  analysis_service / rag_service /     │
└────────────────────────┘        │  profile_service / cleaning_agent /    │
                                    │  auto_analyze_service /               │
                                    │  multi_table_service                  │
                                    │    — pure logic, no FastAPI           │
                                    │      dependency, unit-testable        │
                                    │      directly                         │
                                    │                                        │
                                    │  query_engine.py — shortcut layer,    │
                                    │    prompt building, code cleanup      │
                                    │  llm_provider.py — single Groq API    │
                                    │    client every module calls through  │
                                    │  sandbox.py + sandbox_proxy.py +      │
                                    │    sandbox_limits.py — AST/pattern    │
                                    │    guard, allowlisted pd/np/plt/px    │
                                    │    proxies, timeout + memory cap,     │
                                    │    run in a separate process          │
                                    │  rag_engine.py — parse/chunk/embed/   │
                                    │    store/retrieve, ChromaDB in-memory │
                                    └──────────────────────────────────────┘
```

The same backend also ships a standalone Streamlit UI (`ui.py` / `app.py` in `AST 2.0`) that talks to the service modules directly, with no HTTP hop — useful for local development or running the tool without deploying a frontend at all. An earlier, simpler Streamlit prototype lives in `AST 2.0/legacy_streamlit/` for reference.

---

## Why the sandbox looks the way it does

Every piece of LLM-generated code — pandas analysis, chart generation — runs through:

1. **AST analysis** — blocks imports, dunder access, and a denylist of dangerous calls regardless of encoding or obfuscation tricks.
2. **Allowlisted module proxies** (`sandbox_proxy.py`) — `pd`, `np`, `plt`, `px`, `go` are handed to generated code as capability-limited proxies, not the real modules, so calls like `pd.read_pickle` or `fig.write_html` (which the AST guard can't catch, since they're just ordinary attribute calls) raise `AttributeError` instead of touching the filesystem.
3. **Resource limits** (`sandbox_limits.py`) — a hard wall-clock timeout and, where the OS supports it, a memory cap, enforced by running the exec in a separate process so one runaway query can't hang the whole service.
4. **A curated `__builtins__`** — a safe subset (`len`, `round`, `sum`, etc.) rather than an empty dict, so ordinary analysis code isn't broken as collateral damage.

Multi-table SQL queries go through DuckDB rather than widening the Python sandbox, specifically because DuckDB's own SQL surface has functions (`read_csv`, `httpfs`) that can reach outside the sandbox in ways a keyword denylist can't fully catch — see `multi_table_service.py` for how that's handled.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 15, React 19, TypeScript, Tailwind CSS, Plotly.js |
| File uploads | Vercel Blob (direct browser-to-storage, bypasses the 4.5MB serverless body limit) |
| Backend API | FastAPI (Python) |
| Standalone UI | Streamlit |
| LLM inference | Groq API |
| Data processing | pandas, numpy, DuckDB (multi-table SQL joins) |
| Visualisation | Plotly (primary), matplotlib (fallback, Streamlit path only) |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` (local, CPU) |
| Vector store | ChromaDB (in-memory) |
| Document parsing | pypdf, python-docx, pdfplumber (PDF table extraction) |
| Testing | pytest (backend service modules, sandbox, isolation) |

---

## Getting started

### Backend (`AST 2.0/`)

```bash
cd "AST 2.0"

# CPU-only torch first, avoids conflicts with the LLM extras below
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp .env.example .env
# fill in GROQ_API_KEY, and PYTHON_SERVICE_SECRET if you're running the API

# Option A — standalone Streamlit app
streamlit run app.py

# Option B — FastAPI service for the Next.js frontend
uvicorn api:app --reload --port 8000
```

### Frontend (`AST NextJS/`)

```bash
cd "AST NextJS"
npm install
cp .env.example .env.local
# fill in BLOB_READ_WRITE_TOKEN, PYTHON_SERVICE_URL, PYTHON_SERVICE_SECRET
npm run dev
```

`PYTHON_SERVICE_SECRET` must match on both sides — it's the only thing gating the backend's endpoints on a free-tier deployment with no private networking between services, so every request without a matching `X-Service-Secret` header is rejected with a 401.

### Running tests

```bash
cd "AST 2.0"
pytest                                  # full backend test suite
python eval.py                          # deterministic-component eval harness, no LLM/API key needed
python eval.py --suite security         # sandbox security checks only
```

---

## Project status

- Backend service modules (analysis, cleaning, RAG, profiling, multi-table, auto-analyze) and their FastAPI wrapper are built and unit-tested.
- Frontend CSV and Document Intelligence flows are built and verified end-to-end against a local backend instance.
- Not yet done: real authentication (placeholder identity is used for now), production deployment of either side, and visual parity with the original Streamlit app's density.

See `AST NextJS/README.md` for the full Next.js ↔ FastAPI API contract, and `AST 2.0/README.md` for backend module-by-module detail.
