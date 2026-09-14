"""
rag_service.py — pure logic wrapper around rag_engine.py, no FastAPI
dependency, mirroring analysis_service.py's split from api.py.

Three operations, matching the original Streamlit app's Document
Intelligence tab:
    index_document()  — parse/chunk/embed/store a newly uploaded document
    answer_question()  — RAG Q&A against an already-indexed document
    get_insights()      — auto-extracted ranked insights from the document

Every function returns a plain dict with a `success` key — same
convention as analysis_service.run_analysis(), so api.py's endpoints stay
equally thin for both.

IMPORTANT: the `from rag_engine import ...` line in each function is
INSIDE its try/except, not before it. rag_engine.py transitively imports
streamlit (via llm_provider.py's st.cache_data), so the import itself can
fail with ImportError if that's missing — same as any other failure mode
here, it should degrade to a clean error dict, not crash. (This was a real
bug in an earlier version of this file, caught by test_rag_service.py.)
"""

from __future__ import annotations


def index_document(file_bytes: bytes, filename: str, session_id: str) -> dict:
    """
    Parse -> chunk -> embed -> store. session_id becomes the namespace
    that keeps this document isolated from every other user's documents
    (see rag_engine.py's PER-USER ISOLATION note).
    """
    try:
        from rag_engine import index_document as _index_document, all_deps_ok

        if not all_deps_ok():
            return {
                "success": False,
                "error": (
                    "Document intelligence dependencies (chromadb / "
                    "sentence-transformers) are not installed on the server."
                ),
            }

        meta = _index_document(file_bytes, filename, namespace=session_id)
        return {"success": True, **meta}
    except ValueError as e:
        # Empty/unreadable document, unsupported file type — user-facing,
        # not a server bug.
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Indexing failed: {e}"}


def answer_question(doc_id: str, query: str, doc_name: str) -> dict:
    """RAG Q&A: retrieve relevant chunks, ask the LLM to answer from them."""
    try:
        from rag_engine import rag_answer

        answer = rag_answer(doc_id, query, doc_name)
        return {"success": True, "answer": answer}
    except KeyError:
        # get_collection() raises this when doc_id doesn't exist — most
        # often means the document needs re-indexing (e.g. server restarted
        # since ChromaDB here is in-memory, not persisted to disk).
        return {
            "success": False,
            "error": "Document not found — it may need to be re-uploaded (the index doesn't persist across server restarts).",
        }
    except Exception as e:
        return {"success": False, "error": f"Question answering failed: {e}"}


def get_insights(doc_id: str, doc_name: str) -> dict:
    """Auto-extracted ranked insights, same feature as the original app's
    'Extracts and ranks key insights' — no query needed from the user."""
    try:
        from rag_engine import extract_document_insights

        insights = extract_document_insights(doc_id, doc_name)
        return {"success": True, "insights": insights}
    except KeyError:
        return {
            "success": False,
            "error": "Document not found — it may need to be re-uploaded (the index doesn't persist across server restarts).",
        }
    except Exception as e:
        return {"success": False, "error": f"Insight extraction failed: {e}"}
