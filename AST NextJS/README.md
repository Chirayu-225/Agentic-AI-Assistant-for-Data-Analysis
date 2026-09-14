# Analyst Assistant — Web (Next.js)

Next.js port of the original Streamlit app: frontend + backend API routes in
one repo, talking to your existing Python analysis logic as a separate
service.

## What's built vs. what's next

**Built:**
- CSV upload flow, direct browser-to-storage (Vercel Blob), so the 4.5MB
  Vercel serverless function body limit never applies to the actual file.
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
# fill in BLOB_READ_WRITE_TOKEN, PYTHON_SERVICE_URL, PYTHON_SERVICE_SECRET
npm run dev
```

`BLOB_READ_WRITE_TOKEN` comes from your Vercel project's Storage tab, after
creating a Blob store (free tier is enough for this).

## Why uploads work the way they do

Vercel serverless functions (what a Next.js API route becomes when deployed)
have a hard 4.5MB request body limit — not a Next.js limit, a Vercel
platform limit. `app/api/upload/route.ts` never receives the file; it only
issues a signed token, and the browser uploads the actual bytes straight to
Blob storage. `app/api/analyze/route.ts` then receives a small JSON payload
(a file URL, not the file) and forwards it to your Python service.

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
    "fileUrl": "https://....public.blob.vercel-storage.com/....csv",
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

Building any of these means: fetch the file from `fileUrl` (plain HTTP
GET — see the "Blob access" note below). The `X-Service-Secret` header
should be checked first, as a FastAPI dependency, and any request missing
or mismatching it should get a 401 before any of that work happens —
this is the only thing standing between your sandboxed exec / RAG
endpoints and the open internet, given Render's free tier has no private
networking between services.

## Known trade-off: Blob access is `public`

`FileUpload.tsx` uploads with `access: "public"` — the file gets a
long, unguessable URL, but isn't behind real authentication. This is a
security-through-obscurity trade-off (same threat model as an "anyone with
the link" sharing URL), not a solved problem. `access: "private"` is more
correct but requires generating a signed, short-lived download URL server-side
for the Python service to fetch — extra complexity that's reasonable to add
once real auth exists, not before. Worth revisiting when this moves past a
personal project.

## Deployment

- Frontend: Vercel (free Hobby tier — this is the whole point of the Blob
  upload pattern above).
- Python service: Render free tier, as planned earlier — unaffected by any
  of this, still deployed completely separately.
