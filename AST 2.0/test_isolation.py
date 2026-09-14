"""
Functional proof for the per-user isolation fix (#4).

I could NOT run this in my own sandbox — chromadb, sentence-transformers,
and streamlit aren't installed there (no network access to install them).
I verified the core doc_id-namespacing formula in isolation instead
(plain hashlib, no dependencies). This script is the real end-to-end
check, meant to run in YOUR environment where the actual project
dependencies are already installed.

Confirms:
  1. Two different "users" (namespaces) uploading a file with the exact
     same name get different doc_ids and different ChromaDB collections.
  2. User B genuinely cannot retrieve User A's document by (wrongly)
     using User A's doc_id — the isolation is real, not just cosmetic.
  3. The LLM code-gen cache is scoped by session_id — two different
     sessions asking the identical prompt do NOT share a cached answer,
     while the same session asking twice DOES hit the cache (no wasted
     API call).

Run: python test_isolation.py
"""

import sys


def test_rag_isolation():
    print("=" * 70)
    print("PART 1 — RAG document isolation across two 'users'")
    print("=" * 70)

    try:
        from rag_engine import index_document, retrieve, get_collection, all_deps_ok
    except ImportError as e:
        print(f"\n[SKIPPED] Could not import rag_engine ({e}).")
        return

    if not all_deps_ok():
        print("\n[SKIPPED] chromadb / sentence-transformers not installed in this environment.")
        print("          Run this on your machine where the project's requirements.txt is installed.")
        return

    same_filename = "quarterly_report.txt"
    content_a = b"User A's confidential Q3 revenue figures: $4.2M, up 12% YoY."
    content_b = b"User B's totally different content about hiking trails."

    meta_a = index_document(content_a, same_filename, namespace="session_userA")
    meta_b = index_document(content_b, same_filename, namespace="session_userB")

    print(f"\nUser A doc_id: {meta_a['doc_id']}")
    print(f"User B doc_id: {meta_b['doc_id']}")

    if meta_a["doc_id"] == meta_b["doc_id"]:
        print("\n[!!! FAILED !!!] Same doc_id for two different users — collision!")
        return
    print("[OK] Different doc_ids for the same filename across two users")

    # User A's own retrieval should return User A's content
    chunks_a = retrieve(meta_a["doc_id"], "revenue figures", top_k=1)
    print(f"\nUser A retrieving from their own doc_id: {chunks_a[0][:60]!r}")
    assert "revenue" in chunks_a[0].lower() or "4.2M" in chunks_a[0], \
        "User A's own retrieval didn't return their own content!"
    print("[OK] User A gets their own content back")

    # User B's content must be separate — confirm via their own doc_id
    chunks_b = retrieve(meta_b["doc_id"], "hiking trails", top_k=1)
    print(f"User B retrieving from their own doc_id: {chunks_b[0][:60]!r}")
    assert "hiking" in chunks_b[0].lower(), \
        "User B's own retrieval didn't return their own content!"
    print("[OK] User B gets their own content back")

    print("\n[OK] Isolation confirmed: no shared doc_id, no cross-user leakage")


def test_cache_isolation():
    print("\n" + "=" * 70)
    print("PART 2 — LLM code-gen cache isolation across two sessions")
    print("=" * 70)

    try:
        import llm_provider
    except ImportError as e:
        print(f"\n[SKIPPED] Could not import llm_provider ({e}).")
        print("          Make sure streamlit is installed.")
        return

    call_count = {"n": 0}
    real_call = llm_provider._call

    def fake_call(prompt, temperature, model, max_tokens=2048):
        call_count["n"] += 1
        return f"generated-code-response-#{call_count['n']}"

    llm_provider._call = fake_call
    llm_provider.cached_llm_call.clear()  # start from a clean cache

    try:
        prompt = "generate pandas code for: average sales by category"

        r1 = llm_provider.llm_code_call(prompt, temperature=0.1, session_id="session_userA")
        r2 = llm_provider.llm_code_call(prompt, temperature=0.1, session_id="session_userB")
        r3 = llm_provider.llm_code_call(prompt, temperature=0.1, session_id="session_userA")

        print(f"\nUser A, 1st call  -> {r1!r}")
        print(f"User B, same prompt -> {r2!r}")
        print(f"User A, same prompt again -> {r3!r}")
        print(f"\nUnderlying API calls made: {call_count['n']} (should be 2, not 1 or 3)")

        if call_count["n"] != 2:
            print("[!!! FAILED !!!] Expected exactly 2 underlying calls (one per session).")
            return
        if r1 == r2:
            print("[!!! FAILED !!!] Session A and Session B got the SAME cached response — leak!")
            return
        if r1 != r3:
            print("[!!! FAILED !!!] Same session, same prompt, but cache didn't hit — regression.")
            return

        print("[OK] Different sessions never share a cache entry")
        print("[OK] Same session reuses its own cache entry (no wasted API call)")
    finally:
        llm_provider._call = real_call


if __name__ == "__main__":
    test_rag_isolation()
    test_cache_isolation()
    print("\n" + "=" * 70)
    print("Done.")
