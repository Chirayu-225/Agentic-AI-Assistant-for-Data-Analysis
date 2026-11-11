"""
core/rules.py - the decision table: issue -> Operation, with ZERO LLM calls.

resolve()             rules for everything that has an unambiguous answer
fallback_operation()  a conservative LOW-confidence op for issues the rules cannot
                      resolve, so the app still runs end to end when every LLM
                      provider is down (core principle 4)
quick_options()       the one-click alternatives shown on a proposal card
build_proposals()     rules + (optional) LLM answers + fallbacks, dependency-ordered

Rule summary
  nulls > 60%                         drop_column
  nulls, numeric                      impute median   (skew-robust, so no mean rule)
  nulls, categorical / bool           impute mode
  exact duplicate rows                drop_duplicates
  stray whitespace                    strip_whitespace
  case variants of the same value     normalize_case (dominant style wins)
  date-like strings, format certain   parse_dates(format)
  numeric text, not an identifier     cast_type int|float
Everything else (near-duplicate categories, ambiguous dates, garbage in numeric
columns, outliers, constant columns, nulls in free text/dates/IDs) is "unresolved"
and goes to the single batched LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.profiler import DatasetProfile
from core.schemas import Confidence, Issue, Operation, Proposal

NULL_DROP_PCT = 60.0
NULL_DROP_HIGH_PCT = 80.0
NULL_HIGH_CONF_PCT = 5.0
NULL_MEDIUM_CONF_PCT = 30.0

# Dependency order: clean text first, merge categories, fix types, then fill, then drop.
OP_PRIORITY = {
    "strip_whitespace": 10,
    "map_values": 20,
    "normalize_case": 30,
    "parse_dates": 40,
    "cast_type": 50,
    "drop_duplicates": 60,
    "impute": 70,
    "drop_rows_where_null": 75,
    "clip_outliers": 80,
    "drop_column": 90,
}


@dataclass
class RuleOutcome:
    resolved: dict[str, Operation] = field(default_factory=dict)  # issue id -> op
    unresolved: list[Issue] = field(default_factory=list)  # needs the LLM (or a fallback)
    superseded: list[Issue] = field(default_factory=list)  # moot because the column is dropped


def _null_confidence(pct: float) -> Confidence:
    if pct < NULL_HIGH_CONF_PCT:
        return "high"
    return "medium" if pct <= NULL_MEDIUM_CONF_PCT else "low"


def _kind_overrides(issues: list[Issue]) -> dict[str, str]:
    """Columns whose type will change before imputation runs (text->numeric, text->date)."""
    out: dict[str, str] = {}
    for i in issues:
        if i.type == "mixed_types" and i.evidence.get("subtype") == "numeric_as_text" \
                and not i.evidence.get("identifier_like"):
            out[i.column or ""] = "numeric"
        elif i.type == "date_like_strings":
            out[i.column or ""] = "datetime"
    return out


def _rule_for(issue: Issue, profile: DatasetProfile, kinds: dict[str, str]) -> Operation | None:
    col, ev = issue.column, issue.evidence

    if issue.type == "exact_duplicates":
        return Operation(op="drop_duplicates", rationale=f"{issue.affected_rows} rows are exact copies of an earlier row.",
                         confidence="high")

    if issue.type == "whitespace":
        return Operation(op="strip_whitespace", column=col, params={"collapse": True},
                         rationale=f"{issue.affected_rows} value(s) in '{col}' have stray leading, trailing or doubled spaces.",
                         confidence="high")

    if issue.type == "case_inconsistency":
        style = ev.get("dominant_style", "lower")
        return Operation(op="normalize_case", column=col, params={"mode": style},
                         rationale=f"'{col}' spells the same value in different cases; {style} case is the most common style.",
                         confidence="high" if ev.get("dominant_share", 0) >= 0.8 else "medium")

    if issue.type == "date_like_strings" and not ev.get("ambiguous"):
        n_bad, fmt = ev.get("unparseable_count", 0), ev["format"]
        why = (f"every value matches {fmt}." if n_bad == 0
               else f"{n_bad} value(s) do not match {fmt} and would become empty.")
        return Operation(op="parse_dates", column=col, params={"format": fmt},
                         rationale=f"'{col}' holds dates as text; {why}",
                         confidence="high" if n_bad == 0 else "medium")

    if issue.type == "mixed_types" and ev.get("subtype") == "numeric_as_text" and not ev.get("identifier_like"):
        return Operation(op="cast_type", column=col, params={"dtype": ev["suggested_dtype"]},
                         rationale=f"Every value in '{col}' is a number stored as text.", confidence="high")

    if issue.type == "nulls":
        cp = profile.column(col or "")
        pct = cp.null_pct
        if pct > NULL_DROP_PCT:
            return Operation(op="drop_column", column=col,
                             rationale=f"{pct:.0f}% of '{col}' is empty, too little to fill in reliably.",
                             confidence="high" if pct > NULL_DROP_HIGH_PCT else "medium")
        if cp.looks_like_id:
            return None
        kind = kinds.get(col or "", cp.kind)
        conf = _null_confidence(pct)
        if kind == "numeric":
            skew = "" if cp.skew is None else f" (skew {cp.skew:.1f})"
            return Operation(op="impute", column=col, params={"strategy": "median"},
                             rationale=f"{pct:.1f}% of '{col}' is missing; the median is robust to outliers{skew}.",
                             confidence=conf)
        if kind in ("categorical", "bool"):
            return Operation(op="impute", column=col, params={"strategy": "mode"},
                             rationale=f"{pct:.1f}% of '{col}' is missing; the most common value is the safest fill.",
                             confidence=conf)
    return None


def resolve(issues: list[Issue], profile: DatasetProfile) -> RuleOutcome:
    """Apply the decision table. Pure and deterministic."""
    kinds = _kind_overrides(issues)
    outcome = RuleOutcome()
    for issue in issues:
        op = _rule_for(issue, profile, kinds)
        if op is not None:
            outcome.resolved[issue.id] = op
        else:
            outcome.unresolved.append(issue)

    dropped = {o.column for o in outcome.resolved.values() if o.op == "drop_column"}
    if dropped:  # no point cleaning a column that is about to be removed
        keep_resolved: dict[str, Operation] = {}
        for iid, op in outcome.resolved.items():
            issue = next(i for i in issues if i.id == iid)
            if issue.column in dropped and op.op != "drop_column":
                outcome.superseded.append(issue)
            else:
                keep_resolved[iid] = op
        outcome.resolved = keep_resolved
        still: list[Issue] = []
        for issue in outcome.unresolved:
            (outcome.superseded if issue.column in dropped else still).append(issue)
        outcome.unresolved = still
    return outcome


def fallback_operation(issue: Issue, profile: DatasetProfile) -> Operation | None:
    """A cautious, LOW-confidence proposal for an issue the rules could not resolve.
    Returns None when leaving the data alone is the only safe default."""
    col, ev = issue.column, issue.evidence

    if issue.type == "nulls":
        cp = profile.column(col or "")
        if cp.looks_like_id or cp.kind != "text":
            return None
        return Operation(op="impute", column=col, params={"strategy": "constant", "value": "Unknown"},
                         rationale=f"'{col}' is free text with gaps; labelling them 'Unknown' invents no data.",
                         confidence="low")
    if issue.type == "mixed_types" and ev.get("subtype") == "numeric_with_garbage" and ev.get("numeric_share", 0) >= 0.8:
        return Operation(op="cast_type", column=col, params={"dtype": "float", "errors": "coerce"},
                         rationale=f"'{col}' is mostly numbers; the {issue.affected_rows} non-numeric value(s) would become empty.",
                         confidence="low")
    if issue.type == "date_like_strings" and ev.get("ambiguous"):
        return Operation(op="parse_dates", column=col, params={"format": ev["format"]},
                         rationale=f"'{col}' dates are day/month ambiguous; assuming {ev['format']} (day first).",
                         confidence="low")
    if issue.type == "outliers":
        return Operation(op="clip_outliers", column=col, params={"method": "iqr", "k": ev.get("k", 1.5)},
                         rationale=f"{issue.affected_rows} value(s) in '{col}' lie beyond the IQR fences; clip them to the fences.",
                         confidence="low")
    if issue.type == "constant_column":
        return Operation(op="drop_column", column=col,
                         rationale=f"'{col}' has a single value in every row, so it carries no information.",
                         confidence="low")
    if issue.type == "near_duplicate_categories":
        mapping = {v["value"]: c["canonical"] for c in ev.get("clusters", []) for v in c["variants"]}
        if not mapping:
            return None
        return Operation(op="map_values", column=col, params={"mapping": mapping},
                         rationale=f"'{col}' has near-identical spellings; merging them into the most common one.",
                         confidence="low")
    return None


def quick_options(issue: Issue, profile: DatasetProfile) -> dict[str, Operation]:
    """One-click alternatives for a proposal card (labels match the UI buttons)."""
    if issue.type != "nulls" or not issue.column:
        return {}
    col = issue.column
    cp = profile.column(col)

    def mk(op: str, params: dict, label: str) -> Operation:
        return Operation(op=op, column=None if op == "drop_rows_where_null" else col,  # type: ignore[arg-type]
                         params=params, rationale=f"You chose '{label}' for '{col}'.",
                         confidence="medium", source="user")

    opts: dict[str, Operation] = {}
    if cp.kind == "numeric":
        opts["Median"] = mk("impute", {"strategy": "median"}, "Median")
        opts["Mean"] = mk("impute", {"strategy": "mean"}, "Mean")
    opts["Mode"] = mk("impute", {"strategy": "mode"}, "Mode")
    opts["Drop rows"] = mk("drop_rows_where_null", {"subset": [col]}, "Drop rows")
    return opts


def order_operations(ops: list[Operation]) -> list[Operation]:
    """Stable sort into dependency order (see OP_PRIORITY)."""
    return [o for _, o in sorted(enumerate(ops), key=lambda t: (OP_PRIORITY[t[1].op], t[0]))]


def build_proposals(
    issues: list[Issue],
    profile: DatasetProfile,
    llm_ops: dict[str, Operation] | None = None,
) -> list[Proposal]:
    """Rules first; for unresolved issues use `llm_ops` (issue id -> op) when given,
    otherwise the conservative fallback. Result is in dependency order; proposals with
    no operation come last."""
    outcome = resolve(issues, profile)
    superseded = {i.id for i in outcome.superseded}
    llm_ops = llm_ops or {}
    proposals: list[Proposal] = []
    for issue in issues:
        if issue.id in superseded:
            continue
        op = outcome.resolved.get(issue.id) or llm_ops.get(issue.id) or fallback_operation(issue, profile)
        proposals.append(Proposal(id=issue.id, issue=issue, operation=op,
                                  alternatives=quick_options(issue, profile)))
    keyed = sorted(
        enumerate(proposals),
        key=lambda t: (t[1].operation is None, OP_PRIORITY[t[1].operation.op] if t[1].operation else 0, t[0]),
    )
    return [p for _, p in keyed]


def rules_only_proposals(issues: list[Issue], profile: DatasetProfile) -> list[Proposal]:
    """Everything the app can propose with every LLM provider disabled."""
    return build_proposals(issues, profile, llm_ops=None)
