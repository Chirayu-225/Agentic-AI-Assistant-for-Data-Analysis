"""
Functional proof for rag_service.py.

WHAT THIS CAN TEST HERE
    That every function degrades gracefully (returns a clean error dict)
    even in the worst case — zero of chromadb/sentence-transformers/
    streamlit installed, which is exactly this sandbox's situation. This
    matters because rag_service.py's whole design promise is "never raise,
    always return a dict api.py can pass straight through" — this is the
    hardest case to prove wrong, so it's worth proving.

WHAT THIS CANNOT TEST HERE (and why)
    Real indexing/retrieval/answer generation. Importing rag_engine.py at
    all (even just to check all_deps_ok()) pulls in llm_provider.py at
    module load time, which needs streamlit — so this sandbox can't reach
    ANY real behavior here, not even the "dependencies missing" code path
    specifically, since the import fails before that check ever runs.
    RUN THIS ON YOUR MACHINE for real coverage — with chromadb/
    sentence-transformers/streamlit installed, this should actually index
    a real document and answer a real question about it.

Run: python3 test_rag_service.py
"""

from rag_service import index_document, answer_question, get_insights

print("=" * 70)
print("PART 1 — graceful degradation with zero dependencies (this sandbox)")
print("=" * 70)

sample_bytes = b"This is a test document about quarterly revenue figures."

result = index_document(sample_bytes, "test.txt", "session-1")
print(f"\nindex_document -> {result}")
assert result["success"] is False, "should fail gracefully, not raise"
assert "error" in result and result["error"], "should have a clear error message"
print("[OK] index_document degrades gracefully, no crash")

result = answer_question("some-doc-id", "what is the revenue?", "test.txt")
print(f"\nanswer_question -> {result}")
assert result["success"] is False
assert "error" in result and result["error"]
print("[OK] answer_question degrades gracefully, no crash")

result = get_insights("some-doc-id", "test.txt")
print(f"\nget_insights -> {result}")
assert result["success"] is False
assert "error" in result and result["error"]
print("[OK] get_insights degrades gracefully, no crash")

print("\n" + "=" * 70)
print("PART 2 — the real pipeline (run on your machine)")
print("=" * 70)
print("On your machine, with dependencies installed, run something like:")
print("""
    from rag_service import index_document, answer_question

    with open("some_test_file.txt", "rb") as f:
        content = f.read()

    meta = index_document(content, "some_test_file.txt", "session-1")
    print(meta)   # expect success=True, a doc_id, chunk_count > 0

    result = answer_question(meta["doc_id"], "what is this document about?", "some_test_file.txt")
    print(result)  # expect success=True, a real answer string
""")

print("=" * 70)
print("Done.")
