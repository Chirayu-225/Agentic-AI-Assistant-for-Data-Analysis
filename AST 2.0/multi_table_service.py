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


_KEY_LIKE = re.compile(r"(^|_)(id|code|key|sku|no|num|number)$", re.I)


def _link_details(a_df: pd.DataFrame, b_df: pd.DataFrame, on: list[str], complete: bool) -> dict:
    """Cardinality, orphans and key-type agreement for one a<->b link.

    The classic analytics error this guards against is FAN-OUT: joining
    orders (one row per order) to order_items (many rows per order) and
    then summing an order-level column such as shipping_fee, which counts
    each order once per item. Knowing which side is unique is what lets the
    model (and the user) avoid it. Only computed when both tables are fully
    loaded — a large file is only sampled, so its uniqueness is unknown."""
    if not complete:
        return {"kind": "unknown (large file only sampled)", "aUnique": None, "bUnique": None,
                "orphansInA": None, "orphansInB": None, "typeMismatch": False}
    ka = a_df[on].dropna().astype(str)
    kb = b_df[on].dropna().astype(str)
    a_unique = not ka.duplicated().any()
    b_unique = not kb.duplicated().any()
    kind = {(True, True): "one-to-one", (True, False): "one-to-many",
            (False, True): "many-to-one", (False, False): "many-to-many"}[(a_unique, b_unique)]
    a_keys = set(map(tuple, ka.values.tolist()))
    b_keys = set(map(tuple, kb.values.tolist()))
    orphans_in_a = int(sum(1 for k in map(tuple, ka.values.tolist()) if k not in b_keys))
    orphans_in_b = int(sum(1 for k in map(tuple, kb.values.tolist()) if k not in a_keys))
    mismatch = any(
        pd.api.types.is_numeric_dtype(a_df[c]) != pd.api.types.is_numeric_dtype(b_df[c]) for c in on)
    return {"kind": kind, "aUnique": a_unique, "bUnique": b_unique,
            "orphansInA": orphans_in_a, "orphansInB": orphans_in_b, "typeMismatch": mismatch}


def describe_relationships(named_dfs: dict[str, pd.DataFrame], partial: frozenset = frozenset()) -> dict:
    """Which tables can be joined to which, worked out from column names
    alone (deterministic — no AI). Two tables are linked when they share
    an ID-like column name (customer_id, order_id, sku...). Groups are the
    sets of tables reachable from one another through such links; a table
    in a group of one can't be joined to anything uploaded.

    Shown to the user with every result and given to the model, so "these
    four tables can't all be joined" comes with WHICH ones can and what
    is missing, instead of an empty result. (Found in review: customers,
    order_items and products uploaded without orders.csv returned 0 rows
    and a misleading "only common key is product_id".)"""
    aliases = list(named_dfs)
    cols = {a: {str(c) for c in named_dfs[a].columns} for a in aliases}
    links = []
    for i, a in enumerate(aliases):
        for b in aliases[i + 1:]:
            on = sorted(c for c in cols[a] & cols[b] if _KEY_LIKE.search(c))
            if on:
                complete = a not in partial and b not in partial
                links.append({"a": a, "b": b, "on": on,
                              **_link_details(named_dfs[a], named_dfs[b], on, complete)})
    parent = {a: a for a in aliases}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for l in links:
        parent[find(l["a"])] = find(l["b"])
    groups: dict[str, list[str]] = {}
    for a in aliases:
        groups.setdefault(find(a), []).append(a)
    group_list = sorted(groups.values(), key=len, reverse=True)
    unlinked = [g[0] for g in group_list if len(g) == 1]
    # For an unlinked table, name the ID columns it has, so the user knows
    # what kind of table would connect it ("needs a table with customer_id").
    id_cols = {a: sorted(c for c in cols[a] if _KEY_LIKE.search(c)) for a in unlinked}
    return {
        "links": links,
        "groups": group_list,
        "allConnected": len(group_list) == 1,
        "unlinked": [{"table": a, "idColumns": id_cols[a]} for a in unlinked],
    }


def _relationships_text(rel: dict) -> str:
    lines = ["\nHow the tables connect (from shared ID-column names, checked on the data):"]
    for l in rel["links"]:
        a, b, on = l["a"], l["b"], ", ".join(l["on"])
        kind = l.get("kind", "")
        lines.append(f"  {a} <-> {b} on {on}: {kind}")
        one, many = (a, b) if kind == "one-to-many" else (b, a) if kind == "many-to-one" else (None, None)
        if one:
            lines.append(
                f"    FAN-OUT: each {one} row matches many {many} rows. Per-line maths using a {one} "
                f"column is fine (e.g. quantity * unit_price). But a TOTAL of a {one}-level column "
                f"(a value that belongs to the {one} row itself, like an order's shipping fee or a "
                f"customer's credit limit) gets counted once per {many} row — aggregate {many} to the "
                f"{on} level in a subquery first, or take the {one} values from {one} on its own.")
        elif kind == "many-to-many":
            lines.append(f"    MANY-TO-MANY: joining multiplies rows on both sides; aggregate one side to "
                         f"the key first, or say the question can't be answered reliably.")
        if l.get("orphansInA"):
            lines.append(f"    {l['orphansInA']} {a} row(s) have no matching {b} row (an INNER JOIN drops them).")
        if l.get("orphansInB"):
            lines.append(f"    {l['orphansInB']} {b} row(s) have no matching {a} row (an INNER JOIN drops them).")
        if l.get("typeMismatch"):
            lines.append(f"    Key types differ (text vs number): CAST both sides to VARCHAR when joining.")
    if not rel["links"]:
        lines.append("  no two tables share an ID-like column")
    if rel["allConnected"]:
        lines.append("  All tables are connected.")
    else:
        lines.append("  NOT all tables are connected. Groups that can be joined: "
                     + "; ".join(" + ".join(g) for g in rel["groups"]))
    return "\n".join(lines)


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
    ) + _relationships_text(describe_relationships(named_dfs, frozenset(large_aliases)))
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
- If NOT all tables can be joined (see "How the tables connect"), don't
  return an empty result: join the largest group that CAN be joined and
  answer as much as that allows. In REASONING, name each table you left
  out and what it would need to connect (e.g. "customers was left out:
  nothing uploaded links customer_id to order_id — an orders table with
  both columns would").
- Never invent a join on columns that only look similar by accident.
- Respect every FAN-OUT note above: never SUM/AVG a column from the "one"
  side of a one-to-many join without first aggregating the "many" side
  (or deduplicating). Double counting is the most common join error.
- If even that can't answer the question, write a query that returns an
  empty result, and say why in REASONING rather than guessing.
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
    """Skips byte-identical duplicate uploads, runs the query, and attaches
    the relationship map (which tables link, which don't) plus any notes
    to EVERY response — success or failure — so the UI can always show
    why a join did or didn't happen."""
    import hashlib
    notes: list[str] = []
    seen: dict[str, str] = {}
    unique = []
    for ds in datasets:
        digest = hashlib.sha256(ds["csv_bytes"]).hexdigest()
        if digest in seen:
            notes.append(f"'{ds['filename']}' (as '{ds['alias']}') is identical to "
                         f"'{seen[digest]}' and was skipped.")
            continue
        seen[digest] = ds["alias"]
        unique.append(ds)
    info: dict = {}
    out = _run_multi_table_query(unique, query, session_id, info)
    out["notes"] = notes + info.get("notes", [])
    if "relationships" in info:
        out["relationships"] = info["relationships"]
    return out


def _run_multi_table_query(datasets: list[dict], query: str, session_id: str, info: dict) -> dict:
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
        return {"success": False, "error": "At least 2 different datasets are needed for a multi-table query."}

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
            info["relationships"] = describe_relationships(schema_dfs, frozenset(large_aliases))
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

            # One repair attempt, like single-table questions get: show the
            # model its SQL and the exact DuckDB error (e.g. "Binder Error:
            # Referenced table "o" not found" when an alias is used outside
            # the subquery that defines it — the one failure in the join
            # eval). Timeouts aren't retried: a rewrite rarely fixes those.
            retried = False
            if not sbr.success and "exceeded" not in (sbr.error or "").lower():
                repair_prompt = (
                    prompt
                    + "\n\nYOUR PREVIOUS QUERY FAILED.\nPrevious code:\n```python\n" + code + "\n```\n"
                    + "Error: " + str(sbr.error) + "\n\n"
                    + "Rewrite it so it answers the same question and runs. Common causes: a table "
                    + "alias used outside the subquery that defines it, a column that doesn't exist, "
                    + "or a type mismatch in a join (CAST both sides to VARCHAR). Keep the same "
                    + "REASONING + ```python``` format."
                )
                repair_raw = llm_with_fallback(repair_prompt, temperature=0.0, session_id=session_id)
                repair_reasoning, repair_code = _parse_sql_response(repair_raw)
                if repair_code and repair_code.strip() != code.strip():
                    first_error = sbr.error
                    retried = True
                    reasoning, code = repair_reasoning or reasoning, repair_code
                    sbr = run_multi_table_mixed_sql_sandboxed(code, dataframe_tables, file_tables, memory_mb=memory_mb)
                    if not sbr.success:
                        sbr.error = f"{sbr.error} (first attempt: {first_error})"

            if not sbr.success:
                return {
                    "success": False,
                    "error": sbr.error,
                    "reasoning": reasoning,
                    "code": code,
                    "tables": all_aliases,
                }

            if not sbr.produced or not isinstance(sbr.value, pd.DataFrame):
                return {
                    "success": False,
                    "error": "The query did not return a table — try rephrasing the question.",
                    "reasoning": reasoning,
                    "code": code,
                    "tables": all_aliases,
                }

            records, total_rows, truncated = _records(sbr.value)
            return {
                "success": True,
                "reasoning": reasoning,
                "code": code,
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
