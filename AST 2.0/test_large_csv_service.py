"""
Functional proof for the large-CSV path: sandbox_limits.run_large_csv_sql_sandboxed()
and file_reader.should_use_large_csv_path().

WHY THIS FILE EXISTS
    test_sandbox_limits.py covers run_sandboxed() (the pandas path) and
    test_multi_table_service.py covers run_sql_sandboxed() (the in-memory
    multi-DataFrame path). Neither exercises run_large_csv_sql_sandboxed(),
    which is a third, distinct code path added specifically so a big CSV
    never gets materialized as a pandas DataFrame anywhere — see that
    function's docstring in sandbox_limits.py. Shipping that path without
    a test in this style was the gap this file closes.

WHAT THIS PROVES, END TO END, WITH NO MOCKING OF THE SANDBOX ITSELF
    - should_use_large_csv_path() actually flags files above the 50MB
      threshold, and only CSV/TXT — not Excel (file_reader.py is explicit
      that XLSX has no out-of-core equivalent; this test locks in that
      exact scoping so a future change can't silently widen or narrow it
      without a red test).
    - A real query against a real file ON DISK, executed through the
      FULL real pipeline: subprocess isolation, the AST/pattern guard,
      DuckDB's own CSV engine loading the file (not pandas.read_csv) —
      not a shortcut, and not mocked.
    - The loader connection (the one that actually reads the file off
      disk) is a throwaway, trusted-code-only connection that generated
      code never touches — only the pre-loaded Arrow table, via a
      SEPARATE, locked-down connection, ever reaches generated code. The
      three attack vectors below are the same three proven against
      run_sql_sandboxed() in test_multi_table_service.py, re-run here
      against this newer, file-path-based entry point specifically
      because "the multi-table path is safe" was never a claim about
      this path — each entry point creates its own DuckDB connection and
      has to be checked on its own:
        1. Malicious SQL TEXT trying to read a local file — caught by
           the QUERY connection's enable_external_access=False at the
           engine level (the loader connection's own permissiveness is
           irrelevant here: generated code never has a reference to it).
        2. Generated code trying `.connect()` to escape the wrapper —
           caught by SafeDuckDBConnection's attribute allowlist.
        3. Generated code trying a bare `import duckdb` to get a fresh,
           unrestricted connection — caught by sandbox.py's existing
           blanket import ban.
    - A nonexistent file path fails cleanly (a normal SandboxResult
      error), not with an unhandled exception escaping the worker.

Run: python3 test_large_csv_service.py
"""

import os
import tempfile

import pandas as pd

from file_reader import should_use_large_csv_path, LARGE_FILE_THRESHOLD_BYTES
from sandbox_limits import run_large_csv_sql_sandboxed

print("=" * 70)
print("PART 1 — should_use_large_csv_path() scoping")
print("=" * 70)

small_csv = b"a,b\n1,2\n"
big_csv = b"a,b\n" + b"1,2\n" * (LARGE_FILE_THRESHOLD_BYTES // 4 + 1)

cases = [
    (small_csv, "data.csv", False, "small CSV stays on the normal pandas path"),
    (big_csv, "data.csv", True, "CSV over the 50MB threshold routes to the large-file path"),
    (big_csv, "data.txt", True, "TXT over the threshold also routes to the large-file path"),
    (big_csv, "data.xlsx", False, "XLSX NEVER routes here, regardless of size — no out-of-core engine exists for it (see file_reader.py)"),
]
for file_bytes, filename, expected, label in cases:
    result = should_use_large_csv_path(file_bytes, filename)
    status = "OK" if result == expected else f"MISMATCH (got {result!r}, expected {expected!r})"
    print(f"  [{status}] {label}")
    assert result == expected, f"should_use_large_csv_path({filename!r}, len={len(file_bytes)}) expected {expected}, got {result}"
print("[OK] large-file routing is scoped exactly to CSV/TXT above the size threshold")

print("\n" + "=" * 70)
print("PART 2 — a REAL query, through the REAL sandboxed subprocess,")
print("loading a real file on disk via DuckDB's CSV engine (not pandas)")
print("=" * 70)

sales_df = pd.DataFrame({
    "region": ["West", "East", "West", "East", "North"],
    "amount": [100, 250, 150, 300, 400],
})

tmp_path = None
try:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as tmp:
        sales_df.to_csv(tmp, index=False)
        tmp_path = tmp.name

    query_code = """
result = con.sql(\"\"\"
    SELECT region, SUM(amount) as total
    FROM data
    GROUP BY region
    ORDER BY total DESC
\"\"\").df()
"""
    sbr = run_large_csv_sql_sandboxed(query_code, tmp_path, alias="data")
    print(f"\nsuccess={sbr.success}")
    if sbr.success:
        print(sbr.value)
    assert sbr.success, f"real large-csv query should succeed: {sbr.error}"
    assert list(sbr.value["region"]) == ["East", "North", "West"], f"unexpected aggregation: {sbr.value}"
    print("[OK] real query executes correctly end-to-end, loaded via DuckDB's CSV engine off disk")
finally:
    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)

print("\n" + "=" * 70)
print("PART 3 — three attack vectors, three different layers catching each")
print("(re-verified against THIS entry point — not assumed from the")
print("multi-table path's own test)")
print("=" * 70)

tmp_path = None
try:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as tmp:
        sales_df.to_csv(tmp, index=False)
        tmp_path = tmp.name

    attacks = {
        "malicious SQL text (read local file)": (
            'result = con.sql("SELECT * FROM read_csv(\'/etc/passwd\')").df()',
            "Permission",  # DuckDB's own enable_external_access error, on the QUERY connection
        ),
        ".connect() bypass attempt": (
            "raw_con = con.connect()\nresult = raw_con.sql('SELECT 1').df()",
            "blocked in the sandbox",  # SafeDuckDBConnection's attribute allowlist
        ),
        "raw import duckdb bypass attempt": (
            "import duckdb\nraw = duckdb.connect()\nresult = raw.sql('SELECT 1').df()",
            "import 'duckdb' blocked",  # sandbox.py's existing AST-level import ban
        ),
    }
    for name, (code, expected_fragment) in attacks.items():
        sbr = run_large_csv_sql_sandboxed(code, tmp_path, alias="data")
        print(f"\n  {name}:")
        print(f"    success={sbr.success} | error={sbr.error}")
        assert not sbr.success, f"{name} should have been blocked but succeeded!"
        assert expected_fragment in sbr.error, f"{name}: expected '{expected_fragment}' in error, got: {sbr.error}"
    print("\n[OK] all 3 attack vectors blocked on the large-csv entry point too, each by its own distinct layer")
finally:
    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)

print("\n" + "=" * 70)
print("PART 4 — a nonexistent file path fails cleanly, not with an")
print("unhandled exception escaping the worker process")
print("=" * 70)

sbr = run_large_csv_sql_sandboxed(
    "result = con.sql('SELECT 1').df()",
    "/tmp/this-file-does-not-exist-4f2a9c.csv",
    alias="data",
)
print(f"\nsuccess={sbr.success}")
print(f"error={sbr.error}")
assert not sbr.success, "a missing file should fail, not silently succeed"
assert sbr.error, "a missing file should return a non-empty, readable error message"
print("[OK] missing file fails cleanly with a real error, not a hang or a crash")

print("\n" + "=" * 70)
print("Done.")
