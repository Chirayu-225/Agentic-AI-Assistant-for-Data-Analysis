"""
fill_service.py — the "Fill missing values" button. Takes the session's working table (the agent's, if it has run;
otherwise the rule-cleaned table), fills what fill_missing.py allows, and keeps the result as the working table so the
questions that follow use it. Deterministic, no model.
"""
import pandas as pd

from fill_missing import fill_derived, fill_missing


def _stats(df: pd.DataFrame) -> dict:
    return {"rows": len(df), "columns": len(df.columns), "nullValues": int(df.isna().sum().sum())}


def run_fill(content: bytes, filename: str, session_id: str, file_url: str, unknown: bool = False,
             unknown_columns=None, median: bool = False) -> dict:
    import cache_layer
    import file_reader
    import session_cache
    from cleaning import rule_based_clean

    hit = session_cache.get(session_id, file_url) if session_id and file_url else None
    if hit is not None:
        base = hit.cleaned_df
    else:
        raw = file_reader.read_tabular_file(content, filename)
        base, _ = cache_layer.cached_rule_clean(raw, content, rule_based_clean)

    filled_df, report = fill_missing(base, text_placeholder=bool(unknown), placeholder_columns=unknown_columns,
                                     median=median)
    filled_df, derived = fill_derived(filled_df)
    if derived:
        report["filled"].extend(derived)
        report["total_filled"] += sum(d["count"] for d in derived)
    total = report["total_filled"]
    if total and session_id and file_url:
        session_cache.put(session_id, file_url, filename, content, filled_df)
    inferred = sum(f["count"] for f in report["filled"] if f["method"] == "inferred")
    computed = sum(f["count"] for f in report["filled"] if f["method"] == "derived")
    medians = sum(f["count"] for f in report["filled"] if f["method"] == "median")
    labelled = sum(f["count"] for f in report["filled"] if f["method"] == "text")
    if total:
        parts = []
        if inferred:
            parts.append(f"{inferred:,} from other rows that prove the value")
        if computed:
            parts.append(f"{computed:,} numbers computed exactly from other columns")
        if medians:
            parts.append(f"{medians:,} numbers with their column median")
        if labelled:
            parts.append(f"{labelled:,} text blanks labelled \"Unknown\"")
        message = "Filled " + ", ".join(parts) + ". The filled table is used for your questions."
    else:
        message = "Nothing to fill from the data: the remaining blanks have no evidence in the file."
    return {"success": True, "filename": filename, "before": _stats(base), "after": _stats(filled_df),
            "filled": report["filled"], "skipped": report["skipped"], "remainingText": report["remaining_text"],
            "message": message}
