"""
multi_table_service.py — natural-language questions across several
uploaded datasets at once, joined via real SQL (DuckDB).

WHY THIS EXISTS
    Every other query feature in this app answers a question about ONE
    dataset. This is for "how does X in file A relate to Y in file B" —
    genuinely needs a join, which pandas can do but SQL expresses far
    more naturally (and is what "SQL-esque" queries actually means).

WHY DUCKDB *INSIDE* THE EXISTING SANDBOX, NOT A NEW EXECUTION PATH
    The two options considered: (a) widen the Python sandbox so
    generated code can hold several named DataFrames and write pandas
    joins itself, or (b) run generated SQL text through DuckDB. SQL is
    the better fit for "SQL-esque" — but a naive version of (b) (just
    execute whatever SQL text the LLM writes) has its own real risk:
    DuckDB's SQL syntax itself has built-in functions
    (read_csv/read_parquet, the httpfs extension) that can read
    arbitrary local files or even fetch from a URL — a text-level
    keyword denylist trying to catch every way SQL could invoke those
    is fragile and easy to route around with creative phrasing.

    DuckDB has a real, engine-level answer to this:
    `enable_external_access=False` on the connection, verified (not
    assumed) to block read_csv/httpfs/ATTACH-to-file/INSTALL regardless
    of how the SQL is phrased, while registered-DataFrame joins keep
    working normally. Running that connection INSIDE the existing
    sandboxed subprocess (sandbox_limits.run_sql_sandboxed) adds a
    second, independent layer on top for free: the existing AST guard
    (sandbox.py) unconditionally blocks EVERY `import` statement, so
    generated code can't `import duckdb` itself to get a fresh,
    unrestricted connection — the only DuckDB connection it can ever
    touch is the one this module locks down before generated code sees
    a single line of it (sandbox_proxy.SafeDuckDBConnection wraps that
    connection down to just `.sql()`, nothing else).

WHY NOT AUTOMATIC JOIN-KEY DETECTION BEYOND "SAME COLUMN NAME"
    _find_likely_join_keys() below only looks for identically-named
    columns across tables (e.g. "customer_id" in both `orders` and
    `customers`) — a real but deliberately narrow heuristic. It's a
    hint handed to the LLM, not a decision made for it: the LLM still
    reads full schemas and can join on differently-named columns if the
    question calls for it. Widening this heuristic (fuzzy name
    matching, type-based guessing) risks confidently suggesting a WRONG
    join key, which is worse than suggesting none.
"""

from __future__ import annotations

import io
import os
import re
import tempfile

import numpy as np
import pandas as pd

from cleaning import rule_based_clean
from file_reader import read_tabular_file, should_use_large_csv_path, MAX_RESULT_ROWS


def _json_safe(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if not np.isnan(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _records(df: pd.DataFrame, max_rows: int = MAX_RESULT_ROWS) -> tuple[list[dict], int, bool]:
    """Returns (records, total_row_count, truncated) — same cap and same
    rationale as analysis_service._dataframe_to_records: a join can fan
    out to far more rows than either input table had, so this is at
    least as likely to matter here as on the single-dataset path."""
    total = len(df)
    truncated = total > max_rows
    if truncated:
        df = df.head(max_rows)
    df = df.where(pd.notnull(df), None)
    records = [{k: _json_safe(v) for k, v in row.items()} for row in df.to_dict(orient="records")]
    return records, total, truncated


def sanitize_alias(raw_alias: str, fallback: str) -> str:
    """Table aliases become real SQL identifiers (used directly in
    con.register() and referenced in generated SQL) — must be
    alphanumeric/underscore only and not start with a digit, regardless
    of what the frontend sent. Never trust a name crossing this
    boundary unchecked, even though the frontend already tries to
    generate sane ones from filenames."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", raw_alias).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"t_{cleaned}" if cleaned else fallback
    return cleaned.lower()


def _find_likely_join_keys(named_dfs: dict[str, pd.DataFrame]) -> list[str]:
    """Columns appearing (by exact name) in 2+ tables — a hint, not a
    decision. See this module's docstring for why this stays narrow."""
    column_to_tables: dict[str, list[str]] = {}
    for alias, df in named_dfs.items():
        for col in df.columns:
            column_to_tables.setdefault(col, []).append(alias)
    return [
        f"{col}: appears in {', '.join(tables)}"
        for col, tables in column_to_tables.items()
        if len(tables) > 1
    ]


def _build_schema_text(named_dfs: dict[str, pd.DataFrame], large_aliases: frozenset[str] = frozenset()) -> str:
    """`named_dfs` may hold a SAMPLE DataFrame (not the real, full table)
    for any alias in `large_aliases` — see run_multi_table_query's large-
    file branch. len(df) would silently print the sample's row count as
    if it were the real one in that case, so it's called out explicitly
    instead of just being wrong."""
    sections = []
    for alias, df in named_dfs.items():
        cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
        # AI SECURITY: real sample rows -> prompt. Withhold personal-named
        # columns here; the whole schema block is redacted + fenced below.
        from pii_redaction import mask_personal_columns
        sample = mask_personal_columns(df.head(3)).to_dict(orient="records")
        row_count_text = (
            f"large file, exact row count not loaded — schema/sample only"
            if alias in large_aliases else f"{len(df)} rows"
        )
        sections.append(
            f"TABLE '{alias}' ({row_count_text}):\n"
            f"  columns: {cols}\n"
            f"  sample rows: {sample}"
        )

    join_keys = _find_likely_join_keys(named_dfs)
    join_key_text = (
        "\nLikely join keys (identical column name across tables):\n  " + "\n  ".join(join_keys)
        if join_keys else ""
    )
    from pii_redaction import redact_pii_logged
    from prompt_injection_guard import wrap_untrusted
    text = "\n\n".join(sections) + join_key_text
    return wrap_untrusted(
        redact_pii_logged(text, context="multi-table schema and sample rows"),
        source_label="multi-table schema and sample rows",
    )


def build_multi_table_prompt(schema_text: str, query: str) -> str:
    return f"""You are a data analyst answering a question that requires
querying — and likely joining — multiple datasets, using SQL
(DuckDB syntax, which is close to standard SQL).

{schema_text}

User's question: {query}

Write ONE SQL query answering this question. Respond in exactly this
format — a one-sentence reasoning line, then ONE python code block:

REASONING: <which tables you're using and why, one sentence>

```python
result = con.sql(\"\"\"
<your SQL query here>
\"\"\").df()
```

Rules:
- Only reference tables and columns listed above — never invent one.
- The only operation available is con.sql(...).df() — nothing else.
- Prefer explicit JOIN ... ON clauses over implicit/comma joins.
- If the tables available genuinely can't answer this question, write
  a query that returns an empty result with a comment explaining why,
  rather than guessing at data that isn't there.
"""


def _parse_sql_response(raw: str) -> tuple[str, str]:
    """Splits the response into (reasoning, code) — same REASONING +
    ```python block pattern used by cleaning_agent.py and
    auto_analyze_service.py, so no new parsing convention to learn."""
    code_match = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    code = code_match.group(1).strip() if code_match else ""

    reasoning_match = re.search(r"REASONING\s*:\s*(.+)", raw.replace("**", ""))
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    return reasoning, code


def run_multi_table_query(datasets: list[dict], query: str, session_id: str) -> dict:
    """
    datasets: [{"alias": str, "csv_bytes": bytes, "filename": str}, ...]

    Each dataset is routed individually, on its own size — a join isn't
    "small" or "large" as a whole, one table in it can be either:
      - CSV/TXT over file_reader.LARGE_FILE_THRESHOLD_BYTES: never
        pd.read_csv()'d. Written to a temp file and handed to the sandbox
        as a file path — sandbox_limits.run_multi_table_mixed_sql_sandboxed
        loads it via DuckDB's own CSV engine inside the worker, exactly
        like the single-dataset large-file path
        (sandbox_limits.run_large_csv_sql_sandboxed). Before this, EVERY
        dataset in a join went through read_tabular_file() regardless of
        size, so one large table in a join hit the same ceiling the
        large-file path exists to remove from single-dataset analysis —
        this was the actual gap, not something this function's small-file
        branch below ever had.
      - Everything else (small CSV/TXT, or any XLSX/XLS — Excel has no
        out-of-core engine here, see file_reader.py's module docstring,
        so it always takes this branch regardless of size): read fully
        via read_tabular_file() and rule-cleaned, same as any single-CSV
        upload, exactly as before.

    Rule-based cleaning only runs on the small-file branch, deliberately
    consistent with analysis_service._run_large_csv_analysis, which
    doesn't run it either: cleaning here means rule_based_clean() over a
    full pandas DataFrame, which is exactly the cost the large-file
    branch exists to avoid paying. Not a silent gap — a large table
    joining alongside a cleaned small one may still have the raw currency
    symbols/inconsistent categories rule_based_clean() would otherwise
    fix; a real, documented trade-off (see file_reader.py's own README-
    referenced "Known Limitations"), not something to paper over here.
    """
    if len(datasets) < 2:
        return {"success": False, "error": "At least 2 datasets are needed for a multi-table query."}

    dataframe_tables: dict[str, pd.DataFrame] = {}
    file_tables: dict[str, str] = {}          # alias -> temp file path, large CSV/TXT only
    schema_dfs: dict[str, pd.DataFrame] = {}  # alias -> full df (small) or small SAMPLE df (large)
    large_aliases: set[str] = set()
    tmp_paths: list[str] = []
    used_aliases: set[str] = set()

    try:
        for i, ds in enumerate(datasets):
            alias = sanitize_alias(ds["alias"], fallback=f"table_{i+1}")
            # Guard against two datasets sanitizing to the same alias
            # (e.g. "Orders!" and "orders?" both -> "orders") silently
            # overwriting one another.
            original_alias = alias
            suffix = 2
            while alias in used_aliases:
                alias = f"{original_alias}_{suffix}"
                suffix += 1
            used_aliases.add(alias)

            csv_bytes, filename = ds["csv_bytes"], ds["filename"]

            if should_use_large_csv_path(csv_bytes, filename):
                ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "csv"
                try:
                    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                        tmp.write(csv_bytes)
                        tmp_path = tmp.name
                    tmp_paths.append(tmp_path)

                    # Schema-only sample — same bounded, pandas-based
                    # technique as analysis_service._run_large_csv_analysis;
                    # the real query still runs against the whole file,
                    # via file_tables below, never this sample.
                    try:
                        sample_df = pd.read_csv(io.BytesIO(csv_bytes), nrows=20)
                    except UnicodeDecodeError:
                        sample_df = pd.read_csv(io.BytesIO(csv_bytes), nrows=20, encoding="latin1")
                except Exception as e:
                    return {"success": False, "error": f"Could not read '{filename}': {e}"}

                file_tables[alias] = tmp_path
                schema_dfs[alias] = sample_df
                large_aliases.add(alias)
            else:
                try:
                    df = read_tabular_file(csv_bytes, filename)
                    from cache_layer import cached_rule_clean
                    df, _ = cached_rule_clean(df, csv_bytes, rule_based_clean)
                except Exception as e:
                    return {"success": False, "error": f"Could not parse '{filename}': {e}"}
                dataframe_tables[alias] = df
                schema_dfs[alias] = df

        try:
            from query_engine import llm_with_fallback
            from sandbox_limits import run_multi_table_mixed_sql_sandboxed, DEFAULT_MEMORY_MB, LARGE_FILE_MEMORY_MB

            all_aliases = list(schema_dfs.keys())  # preserves original dataset order across both branches
            schema_text = _build_schema_text(schema_dfs, large_aliases=frozenset(large_aliases))
            prompt = build_multi_table_prompt(schema_text, query)
            raw = llm_with_fallback(prompt, temperature=0.1, session_id=session_id)
            reasoning, code = _parse_sql_response(raw)

            if not code:
                print(f"[multi-table] no code block parsed. Raw response:\n{raw}", flush=True)
                return {
                    "success": False,
                    "error": "Could not generate a SQL query for this question — try again.",
                    "tables": all_aliases,
                }

            # run_multi_table_mixed_sql_sandboxed's own default memory_mb
            # is LARGE_FILE_MEMORY_MB — right when file_tables is
            # non-empty, but a join with no large files at all is
            # ordinary, small-file work that deserves the SAME tight
            # DEFAULT_MEMORY_MB safety net run_sql_sandboxed always used
            # for it — passed explicitly here rather than silently
            # widening the ceiling for every join just because this
            # function is also able to handle large ones.
            memory_mb = LARGE_FILE_MEMORY_MB if file_tables else DEFAULT_MEMORY_MB
            sbr = run_multi_table_mixed_sql_sandboxed(code, dataframe_tables, file_tables, memory_mb=memory_mb)

            if not sbr.success:
                return {
                    "success": False,
                    "error": sbr.error,
                    "reasoning": reasoning,
                    "tables": all_aliases,
                }

            if not sbr.produced or not isinstance(sbr.value, pd.DataFrame):
                return {
                    "success": False,
                    "error": "The query did not return a table — try rephrasing the question.",
                    "reasoning": reasoning,
                    "tables": all_aliases,
                }

            records, total_rows, truncated = _records(sbr.value)
            return {
                "success": True,
                "reasoning": reasoning,
                "tables": all_aliases,
                "resultType": "table",
                "resultData": records,
                # rowCount stays "rows actually in resultData" (matches what
                # MultiTableTab.tsx already renders as "{rowCount} ROW(S)" —
                # no frontend change needed for this to stay accurate even
                # once a join fans out past MAX_RESULT_ROWS). totalRowCount/
                # truncated are new, additive fields for a frontend that
                # wants to show "X of Y" later.
                "rowCount": len(records),
                "totalRowCount": total_rows,
                "truncated": truncated,
                "columns": list(sbr.value.columns),
            }
        except Exception as e:
            return {"success": False, "error": f"Multi-table query failed: {e}"}
    finally:
        # Large-file datasets each got their own temp file above (mirrors
        # analysis_service._run_large_csv_analysis's own cleanup) — clean
        # up regardless of which return path was taken, success or error.
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
