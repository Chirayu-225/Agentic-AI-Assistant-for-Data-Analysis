"""
rag_engine.py — RAG pipeline for unstructured document intelligence.

Pipeline:
  1. Parse   — extract raw text from PDF / DOCX / TXT
  2. Chunk   — split into overlapping passages
  3. Embed   — encode chunks with a local sentence-transformer model
  4. Store   — keep vectors in an in-memory ChromaDB collection
  5. Retrieve — find top-k most relevant chunks for a query
  6. Generate — pass retrieved chunks to Qwen via Ollama for an answer

Only retrieved passage text is sent to the LLM API (Groq) at query time —
parsing, chunking, and embedding all happen locally with no network call.

PER-USER ISOLATION
    doc_id used to be a bare hash of the filename, and _collections/
    _chroma_client are module-level (process-wide) state. In a single-user
    local app that's invisible. In a multi-user deployment it means two
    people uploading a file with the same name silently overwrite each
    other's index, and — more seriously — anyone who can guess or observe
    a filename can query someone else's document, since nothing scopes a
    collection to the user who created it.

    The fix: every caller now passes a `namespace` (the caller's Streamlit
    session_id, or eventually a real authenticated user_id) into
    index_document(), which folds it into the doc_id hash. Once doc_id is
    namespaced, every other function here (retrieve, get_collection,
    rag_answer, extract_document_insights) needs no change at all — they
    only ever receive the doc_id the caller already produced, so the
    isolation is enforced at the one point documents get created, not
    scattered across every read path.
"""

import re
import hashlib
import base64
from pathlib import Path
from typing import Optional

# ── Optional heavy deps — guarded so import errors are clear ──────────────────
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

from llm_provider import llm_chat_call


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_dependencies() -> dict[str, bool]:
    return {
        "chromadb":             CHROMA_AVAILABLE,
        "sentence-transformers": ST_AVAILABLE,
        "pypdf":                PYPDF_AVAILABLE,
        "python-docx":          DOCX_AVAILABLE,
    }


def all_deps_ok() -> bool:
    return all(check_dependencies().values())


# ─────────────────────────────────────────────────────────────────────────────
# 1. PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF byte stream."""
    if not PYPDF_AVAILABLE:
        raise ImportError("pypdf not installed. Run: pip install pypdf")
    from io import BytesIO
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i + 1}]\n{text.strip()}")
    return "\n\n".join(pages)


def parse_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX byte stream."""
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx not installed. Run: pip install python-docx")
    from io import BytesIO
    doc = DocxDocument(BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def parse_txt(file_bytes: bytes) -> str:
    """Decode a plain-text byte stream."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def parse_document(file_bytes: bytes, filename: str) -> str:
    """Route to the correct parser based on file extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_bytes)
    elif ext == ".docx":
        return parse_docx(file_bytes)
    elif ext in (".txt", ".md", ".rst", ".log", ".csv"):
        return parse_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: PDF, DOCX, TXT, MD")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CHUNKER
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[dict]:
    """
    Split text into overlapping chunks for embedding.
    Tries to split on paragraph/sentence boundaries first,
    falls back to hard word-count split.

    Returns list of {"id": str, "text": str, "index": int}.
    """
    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    # Split into paragraphs first
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks    = []
    current   = []
    cur_words = 0

    for para in paragraphs:
        words = para.split()
        if cur_words + len(words) <= chunk_size:
            current.append(para)
            cur_words += len(words)
        else:
            if current:
                chunks.append(" ".join(current))
            # If single paragraph > chunk_size, hard-split it
            if len(words) > chunk_size:
                for start in range(0, len(words), chunk_size - overlap):
                    chunk = " ".join(words[start : start + chunk_size])
                    if chunk.strip():
                        chunks.append(chunk)
                current   = []
                cur_words = 0
            else:
                # Start new chunk with overlap from previous
                overlap_text = chunks[-1].split()[-overlap:] if chunks else []
                current   = [" ".join(overlap_text), para] if overlap_text else [para]
                cur_words = len(" ".join(current).split())

    if current:
        chunks.append(" ".join(current))

    return [
        {
            "id":    hashlib.md5(f"{i}:{c[:60]}".encode()).hexdigest(),
            "text":  c,
            "index": i,
        }
        for i, c in enumerate(chunks)
        if c.strip()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. EMBEDDING MODEL (lazy-loaded singleton)
# ─────────────────────────────────────────────────────────────────────────────

_embed_model: Optional[object] = None

def get_embed_model():
    """
    Lazy-load the embedding model once and cache it in memory.
    Uses all-MiniLM-L6-v2 (~90 MB) — fast, CPU-friendly, good quality.
    """
    global _embed_model
    if _embed_model is None:
        if not ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers"
            )
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """Encode chunk texts into dense vectors."""
    model  = get_embed_model()
    texts  = [c["text"] for c in chunks]
    vecs   = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vecs.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 4. VECTOR STORE (in-memory ChromaDB)
# ─────────────────────────────────────────────────────────────────────────────

_chroma_client = None
_collections: dict[str, object] = {}

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb not installed.\nRun: pip install chromadb"
            )
        _chroma_client = chromadb.Client(
            Settings(anonymized_telemetry=False)
        )
    return _chroma_client


def build_collection(doc_id: str, chunks: list[dict], embeddings: list[list[float]]):
    """
    Create (or recreate) an in-memory ChromaDB collection for this document.
    doc_id should be a stable identifier (e.g. filename hash).
    """
    client = get_chroma_client()

    # Delete existing collection with same name to avoid stale data
    try:
        client.delete_collection(name=doc_id)
    except Exception:
        pass

    collection = client.create_collection(
        name=doc_id,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids        = [c["id"] for c in chunks],
        documents  = [c["text"] for c in chunks],
        embeddings = embeddings,
        metadatas  = [{"index": c["index"]} for c in chunks],
    )
    _collections[doc_id] = collection
    return collection


def get_collection(doc_id: str):
    if doc_id in _collections:
        return _collections[doc_id]
    raise KeyError(f"No collection found for doc_id={doc_id}. Re-index the document.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(doc_id: str, query: str, top_k: int = 5) -> list[str]:
    """
    Embed the query and retrieve the top-k most relevant chunks.
    Returns a list of chunk text strings, ordered by relevance.
    """
    model      = get_embed_model()
    query_vec  = model.encode([query], show_progress_bar=False, convert_to_numpy=True).tolist()
    collection = get_collection(doc_id)
    results    = collection.query(
        query_embeddings = query_vec,
        n_results        = min(top_k, collection.count()),
        include          = ["documents", "distances"],
    )
    return results["documents"][0]   # list of chunk strings


# ─────────────────────────────────────────────────────────────────────────────
# 6. GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_prompt(query: str, chunks: list[str], doc_name: str) -> str:
    context = "\n\n---\n\n".join(
        f"[Passage {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    return f"""You are an expert document analyst. You have been given relevant passages from a document called "{doc_name}".

RETRIEVED PASSAGES:
{context}

USER QUESTION:
{query}

Instructions:
- Answer the question using ONLY the information in the passages above.
- If the answer is not in the passages, say: "This information is not found in the document."
- Be specific. Quote or reference passage numbers where helpful.
- Keep the answer concise and factual. Plain English only.
"""


def build_insight_extraction_prompt(chunks: list[str], doc_name: str) -> str:
    # Use first 8 chunks for overview — enough for a 3B model context window
    sample = "\n\n---\n\n".join(
        f"[Passage {i+1}]\n{chunk}" for i, chunk in enumerate(chunks[:8])
    )
    return f"""You are a senior analyst. Read the following passages from "{doc_name}" and extract the most important insights.

DOCUMENT PASSAGES:
{sample}

Extract 5 key insights from this document, ranked by importance.

Each insight MUST follow this exact format:
[specific finding from the document] → [what it means or what action to take]

Rules:
- Reference specific facts, numbers, names, or claims from the passages
- Be direct — no filler phrases
- If you see risks, flag them first
- If data or statistics are present, include them

Format: 1• finding → implication
"""


def rag_answer(doc_id: str, query: str, doc_name: str,
               top_k: int = 5, temperature: float = 0.2) -> str:
    """Full RAG pipeline: retrieve → build prompt → generate answer via Groq."""
    chunks = retrieve(doc_id, query, top_k=top_k)
    prompt = build_rag_prompt(query, chunks, doc_name)
    return llm_chat_call(prompt, temperature).strip()


def extract_document_insights(doc_id: str, doc_name: str,
                               temperature: float = 0.3) -> str:
    """Pull top chunks and ask the model to extract ranked insights."""
    collection = get_collection(doc_id)
    count      = collection.count()
    # Sample spread of chunks across the document for overview
    all_results = collection.get(include=["documents"])
    all_chunks  = all_results["documents"]
    # Take every Nth chunk for a spread, up to 8
    step   = max(1, len(all_chunks) // 8)
    sample = all_chunks[::step][:8]

    prompt = build_insight_extraction_prompt(sample, doc_name)
    return llm_chat_call(prompt, temperature).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 7. FULL INDEXING PIPELINE (called once per document upload)
# ─────────────────────────────────────────────────────────────────────────────

def index_document(file_bytes: bytes, filename: str, namespace: str = "") -> dict:
    """
    Parse → Chunk → Embed → Store. Returns metadata dict with doc_id,
    chunk_count, char_count, and — for PDFs — any real tables found (see
    table_extraction.py's docstring for why structured data gets pulled
    out and routed to the CSV pipeline instead of being handled by RAG).

    `namespace` should be a stable per-user/per-session identifier
    (e.g. the caller's Streamlit session_id). It's folded into doc_id so
    two different users' documents — even with the identical filename —
    never collide or become queryable by each other. Defaults to "" only
    for backward compatibility with any existing direct callers; every
    real call site should pass a real namespace.
    """
    # Parse
    raw_text = parse_document(file_bytes, filename)
    if not raw_text.strip():
        raise ValueError("Document appears to be empty or unreadable.")

    # Chunk
    chunks = chunk_text(raw_text, chunk_size=400, overlap=80)
    if not chunks:
        raise ValueError("Could not extract any text chunks from the document.")

    # Embed
    embeddings = embed_chunks(chunks)

    # Store — hash of namespace + filename as stable, per-user doc_id
    doc_id = hashlib.md5(f"{namespace}:{filename}".encode()).hexdigest()[:16]
    build_collection(doc_id, chunks, embeddings)

    # Structured-data detection — PDF only, best-effort. A table
    # extraction failure here should never break document indexing
    # itself; the document is still fully usable for RAG Q&A either way.
    tables: list[dict] = []
    if filename.lower().endswith(".pdf"):
        try:
            from table_extraction import extract_tables_from_pdf
            tables = extract_tables_from_pdf(file_bytes, filename)
        except Exception:
            tables = []

    return {
        "doc_id":      doc_id,
        "filename":    filename,
        "char_count":  len(raw_text),
        "chunk_count": len(chunks),
        "word_count":  len(raw_text.split()),
        "tables": [
            {
                "title": t["title"],
                "page": t["page"],
                "row_count": t["row_count"],
                "col_count": t["col_count"],
                "preview": t["preview"],
                # base64 so this survives a plain JSON response —
                # api.py's routes return dicts straight as JSON, and raw
                # bytes aren't JSON-serialisable.
                "csv_base64": base64.b64encode(t["csv_bytes"]).decode("ascii"),
            }
            for t in tables
        ],
    }
