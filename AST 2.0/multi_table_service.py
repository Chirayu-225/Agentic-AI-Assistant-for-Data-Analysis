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

import re

import numpy as np
import pandas as pd

from cleaning import rule_based_clean
from file_reader import read_tabular_file


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


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.where(pd.notnull(df), None)
    return [{k: _json_safe(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


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


def _build_schema_text(named_dfs: dict[str, pd.DataFrame]) -> str:
    sections = []
    for alias, df in named_dfs.items():
        cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
        sample = df.head(3).to_dict(orient="records")
        sections.append(
            f"TABLE '{alias}' ({len(df)} rows):\n"
            f"  columns: {cols}\n"
            f"  sample rows: {sample}"
        )

    join_keys = _find_likely_join_keys(named_dfs)
    join_key_text = (
        "\nLikely join keys (identical column name across tables):\n  " + "\n  ".join(join_keys)
        if join_keys else ""
    )
    return "\n\n".join(sections) + join_key_text


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
    Each file is read + rule-cleaned exactly like any single-CSV
    upload — no different treatment for being part of a join.
    """
    if len(datasets) < 2:
        return {"success": False, "error": "At least 2 datasets are needed for a multi-table query."}

    named_dfs: dict[str, pd.DataFrame] = {}
    used_aliases: set[str] = set()
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

        try:
            df = read_tabular_file(ds["csv_bytes"], ds["filename"])
            df, _ = rule_based_clean(df)
        except Exception as e:
            return {"success": False, "error": f"Could not parse '{ds['filename']}': {e}"}
        named_dfs[alias] = df

    try:
        from query_engine import llm_with_fallback
        from sandbox_limits import run_sql_sandboxed

        schema_text = _build_schema_text(named_dfs)
        prompt = build_multi_table_prompt(schema_text, query)
        raw = llm_with_fallback(prompt, temperature=0.1, session_id=session_id)
        reasoning, code = _parse_sql_response(raw)

        if not code:
            print(f"[multi-table] no code block parsed. Raw response:\n{raw}", flush=True)
            return {
                "success": False,
                "error": "Could not generate a SQL query for this question — try again.",
                "tables": list(named_dfs.keys()),
            }

        sbr = run_sql_sandboxed(code, named_dfs)

        if not sbr.success:
            return {
                "success": False,
                "error": sbr.error,
                "reasoning": reasoning,
                "tables": list(named_dfs.keys()),
            }

        if not sbr.produced or not isinstance(sbr.value, pd.DataFrame):
            return {
                "success": False,
                "error": "The query did not return a table — try rephrasing the question.",
                "reasoning": reasoning,
                "tables": list(named_dfs.keys()),
            }

        return {
            "success": True,
            "reasoning": reasoning,
            "tables": list(named_dfs.keys()),
            "resultType": "table",
            "resultData": _records(sbr.value),
            "rowCount": len(sbr.value),
            "columns": list(sbr.value.columns),
        }
    except Exception as e:
        return {"success": False, "error": f"Multi-table query failed: {e}"}
