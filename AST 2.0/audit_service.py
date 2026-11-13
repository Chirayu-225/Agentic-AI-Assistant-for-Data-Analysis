"""
audit_service.py — the audit trail behind the dashboard.

  build_audit    re-runs the deterministic pipeline on the upload (the reader and the cleaner come from the cache, so this is
                 cheap) and records every change in an audit_ledger.Ledger. scope "clean" = the rule cleaner only;
                 scope "agent" = also the fill, the value repairs, the exact-arithmetic fill and whatever the AI step and
                 the user's answers left in the session's working table.
  audit_page     one page of the change list (filters), for the dashboard.
  audit_csv      the whole change list as CSV.
  remove_duplicates   the user's answer to "keep or remove the identical rows": drops the repeats from the working table.

No model is involved anywhere in here.
"""
from __future__ import annotations

import pandas as pd

import audit_ledger as al

RULE_LABELS = {
    "unit_in_text": "Unit in text converted",
    "percent_scale": "Percent scale fixed",
    "magnitude_outlier": "Decimal shift fixed",
    "impossible_negative": "Impossible negative blanked",
    "future_date": "Future date blanked",
    "end_before_start": "End date before start blanked",
    "mixed_dates": "Mixed date formats unified",
    "swapped_values": "Swapped values corrected",
    "near_duplicate": "Near-duplicate row removed",
}
_ACTION_OF = {"rescaled": "changed", "converted": "changed", "swapped": "changed", "unified": "changed",
              "blanked": "blanked", "dropped": "removed"}


def _key(session_id, file_url, scope):
    return (session_id, file_url, scope)


def _validator_rules(vr: dict):
    rules, ev = {}, {}
    for e in vr.get("entries", []):
        label = RULE_LABELS.get(e.get("rule"), str(e.get("rule")))
        col = e.get("column")
        act = _ACTION_OF.get(e.get("action"), "changed")
        rules[(col, act)] = label
        rules.setdefault(col, label)
        ev[("Value repairs", label, col)] = e.get("evidence", "")
    return rules, ev


def build_audit(content: bytes, filename: str, session_id: str, file_url: str, scope: str = "clean", force: bool = False):
    """-> (ledger, summary), from the store when this exact (session, file, scope) was built a moment ago."""
    import cache_layer
    import session_cache
    from cleaning import rule_based_clean
    from file_reader import read_tabular_file

    scope = "agent" if scope == "agent" else "clean"
    key = _key(session_id, file_url, scope)
    if not force:
        hit = al.store_get(key)
        if hit is not None:
            return hit

    raw = read_tabular_file(content, filename)
    clean, log = cache_layer.cached_rule_clean(raw, content, rule_based_clean)
    ledger = al.Ledger.for_clean(raw, clean)
    ledger.start(clean)
    ledger.evidence.update({("Rule cleaning", None, None): ""})
    # evidence for the cleaner's rules: the log line that names the column
    wanted = {"Placeholder value (sentinel) replaced by blank": "sentinel", "Spelling variant merged": "Merged spelling",
              "Spacing or case standardised": "Merged spelling", "Value could not be read, left blank": "could not"}
    for line in log:
        for col in clean.columns:
            if f"'{col}'" in line:
                for rule, word in wanted.items():
                    if word in line:
                        ledger.evidence.setdefault(("Rule cleaning", rule, str(col)), line.lstrip("✓⚠ ").strip())
    final = clean
    if scope == "agent":
        from fill_missing import fill_derived, fill_missing
        from value_validators import validate_values
        filled, fr = fill_missing(clean)
        fill_rules = {e["column"]: "Filled from other rows that prove the value" for e in fr["filled"]}
        ledger.advance("Fill from evidence", filled, rules=fill_rules,
                       evidence={("Fill from evidence", "Filled from other rows that prove the value", e["column"]): e.get("value", "")
                                 for e in fr["filled"]})
        repaired, vr = validate_values(filled)
        rules, ev = _validator_rules(vr)
        ledger.advance("Value repairs", repaired, rules=rules, removed_reason="Near-duplicate row removed", evidence=ev)
        derived, de = fill_derived(repaired)
        ledger.advance("Exact arithmetic", derived, default_rule="Computed exactly from other columns",
                       evidence={("Exact arithmetic", "Computed exactly from other columns", e["column"]): e.get("value", "")
                                 for e in de})
        final = derived
    work = session_cache.get(session_id, file_url)
    if work is not None and isinstance(work.cleaned_df, pd.DataFrame):
        # the table the questions use: whatever the AI step, the fill buttons and the duplicate choice left in it
        ledger.advance("AI fix and your choices", work.cleaned_df,
                       default_rule="AI-suggested safe change, or your answer", removed_reason="Removed at your request")
        final = work.cleaned_df
    summary = ledger.summary(final, dup_columns=[c for c in clean.columns if c in final.columns])
    summary["scope"] = scope
    summary["filename"] = filename
    ledger.release()
    al.store_put(key, ledger, summary)
    return ledger, summary


def audit_response(content, filename, session_id, file_url, scope="clean", offset=0, limit=100,
                   rule=None, column=None, action=None, q=None, force=False) -> dict:
    ledger, summary = build_audit(content, filename, session_id, file_url, scope, force=force)
    page = ledger.page(offset=offset, limit=limit, rule=rule, column=column, action=action, q=q)
    return {"success": True, "summary": summary, **page}


def audit_csv(content, filename, session_id, file_url, scope="clean") -> str:
    ledger, _ = build_audit(content, filename, session_id, file_url, scope)
    return ledger.to_csv()


def remove_duplicates(content: bytes, filename: str, session_id: str, file_url: str) -> dict:
    """Drop the repeated rows (identical on every original column; the first copy stays) from the working table, keep the
    result as the session's working table, and forget the stored audit so the next one shows the removal."""
    import cache_layer
    import session_cache
    from cleaning import rule_based_clean
    from file_reader import read_tabular_file

    hit = session_cache.get(session_id, file_url)
    if hit is not None:
        base = hit.cleaned_df
    else:
        raw = read_tabular_file(content, filename)
        base, _ = cache_layer.cached_rule_clean(raw, content, rule_based_clean)
    raw_cols = None
    try:
        raw = read_tabular_file(content, filename)
        clean, _ = cache_layer.cached_rule_clean(raw, content, rule_based_clean)
        raw_cols = [c for c in clean.columns if c in base.columns]
    except Exception:  # noqa: BLE001
        raw_cols = list(base.columns)
    dup = base.duplicated(subset=raw_cols, keep="first")
    removed = int(dup.sum())
    if removed:
        new = base[~dup]                                   # index labels are kept: they are how rows are traced
        session_cache.put(session_id, file_url, filename, content, new)
    al.store_drop((session_id, file_url))
    def _stats(d):
        return {"rows": int(len(d)), "columns": int(len(d.columns)), "nullValues": int(d.isna().sum().sum())}
    after_df = base[~dup] if removed else base
    return {"success": True, "removed": removed, "rows": int(len(base) - removed),
            "before": _stats(base), "after": _stats(after_df),
            "message": (f"Removed {removed:,} repeated row{'s' if removed != 1 else ''}; the first copy of each stays. "
                        "The table without them is used for your questions.") if removed else "There were no repeated rows to remove."}
