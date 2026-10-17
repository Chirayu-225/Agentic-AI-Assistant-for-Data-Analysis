# Analyst Assistant — Web (Next.js)

Next.js port of the original Streamlit app: frontend + backend API routes in
one repo, talking to your existing Python analysis logic as a separate
service.

## What's built vs. what's next

**Built:**
- File upload streamed through the app to the Python backend, which keeps the
  file on local disk (no cloud storage, no tokens). Large files are streamed,
  never buffered.
- CSV query panel + result display (table / scalar / chart) — verified
  working end-to-end against a real Python service.
- Document Intelligence tab — upload PDF/DOCX/TXT/MD, ask questions (RAG
  Q&A), or get auto-extracted ranked insights. Same upload/isolation
  pattern as the CSV tab.
- A per-session identifier (`lib/sessionId.ts`) matching the `session_id`
  concept used in `sandbox_proxy.py` / `rag_engine.py` on the Python side.
- The Python FastAPI service this calls (`api.py`, `analysis_service.py`,
  `rag_service.py` — see the Python repo). Not yet deployed anywhere,
  only tested against `localhost:8000`.

**Deliberately not built yet:**
- Real auth (there are `TODO` markers for this — `app/api/upload/route.ts`
  and `lib/sessionId.ts` — both currently use a placeholder identity).
- Visual polish matching the original Streamlit app's density (sidebar
  stats, numbered cards) — functionality was prioritized first.
- Actual deployment (Python service to Render, confirming the
  Vercel-hosted frontend can reach it over the public internet).


## Setup

```bash
npm install
cp .env.example .env.local
# fill in PYTHON_SERVICE_URL and PYTHON_SERVICE_SECRET (must match the backend's .env)
npm run dev
```

That is the whole frontend setup — there is no storage account or token.

## How uploads work

The browser POSTs the raw file to `/api/upload`. That route streams the bytes
on to the Python service's `POST /upload` (never holding the whole file in
memory), and the service stores it in a temp folder on disk and returns an id
like `local://3f9a...`. Every other route (`/api/analyze`, `/api/profile`, ...)
keeps receiving that id as `fileUrl`, and the Python service resolves it back
to the stored file. The backend enforces the size cap (`MAX_UPLOAD_MB`, default
200) and deletes files after `LOCAL_UPLOAD_TTL_HOURS` (default 24).

This replaced Vercel Blob. Blob existed only to get around Vercel's 4.5MB
serverless body limit, which does not apply when running locally. If the app is
ever deployed to Vercel, large uploads would need a storage service again.

## Python API contract

This is implemented in `api.py`/`analysis_service.py`/`rag_service.py` in
the Python repo — documented here so the contract stays the source of
truth if either side changes independently.

### POST /analyze

```
Headers:
  X-Service-Secret: <matches PYTHON_SERVICE_SECRET>
Body:
  {
    "fileUrl": "local://<id returned by POST /upload>",
    "filename": "sales.csv",
    "query": "what are the top 5 rows by sales?",
    "sessionId": "a-uuid-from-the-browser"
  }
Response (200):
  {
    "success": true,
    "resultType": "table" | "scalar" | "chart" | "error",
    "resultData": ...,   // shape depends on resultType, see below
    "error": null,
    "confidence": "HIGH" | "MEDIUM" | "LOW"
  }
```

`resultData` shape by `resultType`:
- `"scalar"` — a plain number or string.
- `"table"` — an array of row objects: `[{"col_a": 1, "col_b": "x"}, ...]`
  (i.e. `df.to_dict(orient="records")`).
- `"chart"` — a Plotly figure as JSON: `{"data": [...], "layout": {...}}`
  (i.e. `json.loads(fig.to_json())` on a Plotly figure — matplotlib
  figures aren't converted; `_shape_result()` returns an explicit
  "not supported yet" error for those rather than something broken).
- `"error"` — `resultData` can be `null`; put the message in `error`.

### POST /rag/index

Parses, chunks, embeds, and stores a document for later Q&A. Call once
right after upload, before the query panel is usable.

```
Body:
  { "fileUrl": "...", "filename": "report.pdf", "sessionId": "..." }
Response (200):
  {
    "success": true,
    "doc_id": "a16charhash",
    "filename": "report.pdf",
    "char_count": 12000,
    "chunk_count": 34,
    "word_count": 2100
  }
```

`doc_id` is namespaced by `sessionId` server-side (see `rag_engine.py`'s
per-user isolation) — the frontend just stores and reuses whatever
`doc_id` comes back, no isolation logic needed on this side.

### POST /rag/query

```
Body:
  { "docId": "...", "filename": "report.pdf", "query": "what are the risks?" }
Response (200):
  { "success": true, "answer": "..." }
```

### POST /rag/insights

Auto-extracted ranked insights, no query needed from the user.

```
Body:
  { "docId": "...", "filename": "report.pdf" }
Response (200):
  { "success": true, "insights": "1• finding → implication\n2• ..." }
```

The frontend (`RagResultDisplay.tsx`) parses that `N•` line format into
styled rank/body pairs — matching `rag_engine.py`'s
`build_insight_extraction_prompt()` output format exactly.

### Common to all four

Building any of these means: load the file named by `fileUrl` (a `local://` id
from `POST /upload`, or any plain HTTP URL). The `X-Service-Secret` header
should be checked first, as a FastAPI dependency, and any request missing
or mismatching it should get a 401 before any of that work happens —
this is the only thing standing between your sandboxed exec / RAG
endpoints and the open internet, given Render's free tier has no private
networking between services.

## Security note on stored files

Uploaded files get a content-hash id and are only reachable through the backend,
which requires the shared `X-Service-Secret` header; ids are validated against a
strict pattern so they cannot be used to read other paths. There is no per-user
access control yet (same as the rest of the app) — add it together with real
authentication.

## Deployment

- The project is built to run locally. Deploying the frontend to Vercel would
  need an external storage service for large uploads (Vercel's 4.5MB request
  limit).
- Python service: Render free tier, as planned earlier — unaffected by any
  of this, still deployed completely separately.
