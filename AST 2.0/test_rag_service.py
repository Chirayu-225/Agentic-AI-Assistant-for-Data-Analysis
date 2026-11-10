"""
test_rag_service.py — rag_service.py: never raises, always returns a dict api.py can pass straight through.

PART 1 (always runs; fast and offline). The dependencies ARE installed here, so this tests the real behaviour of the
failure paths that need no embedding model and no LLM:
    * an unsupported file type and an empty document are refused by index_document with a user-facing message
      (the document is parsed BEFORE any embedding model is loaded);
    * an unknown doc_id gives the "re-upload" message from answer_question and get_insights (the real collection
      lookup raises the KeyError).
  The LLM is replaced by a function that fails the test if it is ever called, so nothing here can reach Groq, and
  the embedding model is replaced by a stub, so nothing is loaded or downloaded.

PART 2 (opt-in): the full index -> answer round trip. It needs the sentence-transformers embedding model (a download
  on first use) and a live LLM, so it only runs with RAG_FULL_TEST=1. Otherwise it prints [SKIPPED].

Run: python test_rag_service.py        (full round trip: set RAG_FULL_TEST=1)
"""
import os
import sys

FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def main():
    from rag_service import index_document, answer_question, get_insights

    full = os.environ.get("RAG_FULL_TEST") == "1"
    if not full:
        # Nothing in part 1 may call the model; if it does, that is a failure, not a Groq request.
        import rag_engine

        def no_llm(*a, **k):
            raise AssertionError("the LLM must not be called in the offline part of this test")
        rag_engine.llm_chat_call = no_llm

        # answer_question embeds the query BEFORE it looks the collection up, so the real embedding model would be
        # loaded (slow, and a download on a fresh machine) just to reach the "unknown doc_id" answer. A stub embedder
        # keeps this offline; the collection lookup that produces the KeyError is the real one.
        class _NoModel:
            def encode(self, texts, **k):
                import numpy as np
                return np.zeros((len(texts), 1))
        rag_engine.get_embed_model = lambda: _NoModel()

    print("=" * 70)
    print("PART 1 — real failure paths, offline")
    print("=" * 70)

    r = index_document(b"MZ\x90\x00 not a document", "payload.exe", "s1")
    check("unsupported file type: success False with a clear message",
          r.get("success") is False and "Unsupported file type" in (r.get("error") or ""), str(r))

    r = index_document(b"   \n\t  ", "blank.txt", "s1")
    check("empty document: success False with a clear message",
          r.get("success") is False and "empty or unreadable" in (r.get("error") or ""), str(r))

    for name, call in (("answer_question", lambda: answer_question("no-such-doc", "what is this?", "x.txt")),
                       ("get_insights", lambda: get_insights("no-such-doc", "x.txt"))):
        r = call()
        check(f"{name}: unknown doc_id gives the re-upload message",
              r.get("success") is False and "Document not found" in (r.get("error") or ""), str(r))

    print()
    print("=" * 70)
    print("PART 2 — full round trip (opt-in: RAG_FULL_TEST=1)")
    print("=" * 70)
    if not full:
        print("[SKIPPED] set RAG_FULL_TEST=1 to index a real document and ask a question "
              "(downloads the embedding model on first use and calls the live LLM).")
    else:
        meta = index_document(b"Quarterly revenue was 4.2 million in Q3, up 12 percent on Q2.", "rev.txt", "s-full")
        check("index_document succeeds with a doc_id and chunks",
              meta.get("success") is True and meta.get("doc_id") and meta.get("chunk_count", 0) > 0, str(meta)[:200])
        if meta.get("success"):
            ans = answer_question(meta["doc_id"], "What was the Q3 revenue?", "rev.txt")
            check("answer_question returns an answer string", ans.get("success") is True and ans.get("answer"), str(ans)[:200])

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("DONE — all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
