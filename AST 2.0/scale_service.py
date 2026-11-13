"""scale_service.py - applies the user's "yes, divide by 1000" answer to the working table (see scale_review.py)."""
from __future__ import annotations

from scale_review import apply_scale_fix


def _stats(df) -> dict:
    return {"rows": int(len(df)), "columns": int(len(df.columns)), "nullValues": int(df.isna().sum().sum())}


def run_scale_fix(content: bytes, filename: str, session_id: str, file_url: str, columns: list[str]) -> dict:
    import audit_ledger as al
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
    new, done = apply_scale_fix(base, [str(c) for c in columns][:50])
    total = sum(d["count"] for d in done)
    if total and session_id and file_url:
        session_cache.put(session_id, file_url, filename, content, new)
        al.store_drop((session_id, file_url))        # the next audit must show these changes
    if total:
        parts = ", ".join(f"{d['count']:,} in '{d['column']}'" for d in done)
        message = f"Divided {total:,} value{'s' if total != 1 else ''} by 1000 ({parts}). The corrected table is used for your questions."
    else:
        message = "Nothing to change: no values in that column still look 1000x too large."
    return {"success": True, "filename": filename, "changed": total, "columns": done,
            "before": _stats(base), "after": _stats(new), "message": message}
