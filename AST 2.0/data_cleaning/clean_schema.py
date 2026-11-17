"""
clean_schema.py — the contracts of the agentic cleaning pipeline.

Everything in the pipeline hangs off four objects, defined here and nowhere else:

  Flag            one row-level finding, keyed by SOURCE coordinate (sheet + the
                  row number the user sees in their own file).
  Proposal        one reviewable change. Detectors emit these today, an LLM
                  planner later; both go through the same validation.
  Evidence packet the ONLY thing an LLM ever sees. Bounded by columns and by a
                  token budget, never by rows.
  DeclaredEffect  what a proposal promises to change. The verifier compares the
                  promise with what actually happened.

Invariants enforced here (not by convention):
  * LLM cost is O(columns), never O(rows): build_packet() has a column cap and a
    hard token budget.
  * Raw values do not reach the LLM: evidence has separate `llm_visible` and
    `user_only` parts, and the packet builder never receives the latter.
  * Every string that enters a packet is sanitised and length-capped (file
    contents are untrusted: a category value can say "ignore previous
    instructions").
  * Only a fixed set of declarative ops exists; params are validated per op.
  * Defaults are not permissions: nothing here applies a change. A proposal only
    carries a default_action for the review screen to pre-select.

Stdlib only, so it can be tested without pandas.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, NamedTuple, Optional


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, ""))
    except ValueError:
        return default


# ── limits ───────────────────────────────────────────────────────────────────
MAX_NAME_LEN = 64
MAX_VALUE_LEN = 32
MAX_PATTERN_LEN = 24
MAX_EXAMPLE_LEN = 80
MAX_EXAMPLES = 5
TOPK_MAX_CARDINALITY = 20      # top-k values are only ever shown for low-cardinality columns
TOPK_MAX_VALUES = 10
MAX_PACKET_COLUMNS = 40        # a 300-column sheet must not blow the token budget
PACKET_TOKEN_BUDGET = 6000
SMALL_TABLE_ROWS = 100         # below this, statistical findings are never pre-selected
MAX_MAP_ENTRIES = 200
CONTRACT_RULE_MAX_FLAG_SHARE = _env_float("CLEAN_RULE_MAX_FLAG_SHARE", 0.10)


# ── enums ────────────────────────────────────────────────────────────────────
class Tier(str, Enum):
    SAFE = "safe"          # lossless normalisation; pre-selected as a group
    REVIEW = "review"      # needs the user's eye; flag-only by default
    HARD = "hard"          # a provable violation; quarantine / blank pre-selected


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    HARD = "hard"


class Basis(str, Enum):
    DETERMINISTIC = "deterministic"
    STATISTICAL = "statistical"
    LLM = "llm"


class ScopeKind(str, Enum):
    CELLS = "cells"
    ROWS = "rows"


class Choice(str, Enum):
    APPLY = "apply"            # make the proposed change
    BLANK = "blank"            # null the suspicious cell(s) instead of guessing (cells only)
    QUARANTINE = "quarantine"  # exclude the row(s) from analysis, reversibly (rows only)
    KEEP = "keep"              # leave as is; the finding stays visible


class OpName(str, Enum):
    CAST = "cast"
    TRIM = "trim"
    PARSE_DATE = "parse_date"
    MAP_VALUES = "map_values"
    IMPUTE = "impute"
    DEDUPE = "dedupe"
    CLIP = "clip"
    QUARANTINE = "quarantine"
    FLAG = "flag"
    BLANK = "blank"


class ReasonCode(str, Enum):
    # structure (spreadsheet layout)
    HEADER_OFFSET = "HEADER_OFFSET"
    MERGED_CELLS = "MERGED_CELLS"
    FOOTER_NOTE = "FOOTER_NOTE"
    TOTAL_ROW = "TOTAL_ROW"
    HIDDEN_ROWS = "HIDDEN_ROWS"
    MULTI_TABLE = "MULTI_TABLE"
    # cell-level
    XL_ERROR_VALUE = "XL_ERROR_VALUE"
    DATE_SERIAL = "DATE_SERIAL"
    FORMULA_OVERRIDDEN = "FORMULA_OVERRIDDEN"
    NUMBER_AS_TEXT = "NUMBER_AS_TEXT"
    SENTINEL_NULL = "SENTINEL_NULL"
    WHITESPACE = "WHITESPACE"
    DATE_AMBIGUOUS = "DATE_AMBIGUOUS"
    CURRENCY_MIXED = "CURRENCY_MIXED"
    # value / category
    CATEGORY_VARIANT = "CATEGORY_VARIANT"
    OUTLIER_ROBUST = "OUTLIER_ROBUST"
    IMPLAUSIBLE_RANGE = "IMPLAUSIBLE_RANGE"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    # row / cross-column
    DUP_EXACT = "DUP_EXACT"
    DUP_NEAR = "DUP_NEAR"
    DATE_END_BEFORE_START = "DATE_END_BEFORE_START"
    RULE_VIOLATION = "RULE_VIOLATION"
    SWAPPED_COLUMNS = "SWAPPED_COLUMNS"


_SCOPE_CHOICES = {
    ScopeKind.ROWS: {Choice.APPLY, Choice.QUARANTINE, Choice.KEEP},
    ScopeKind.CELLS: {Choice.APPLY, Choice.BLANK, Choice.KEEP},
}


# ── sanitising ───────────────────────────────────────────────────────────────
_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_WS = re.compile(r"\s+")


def sanitize_text(value: Any, max_len: int) -> str:
    """Control characters out, whitespace collapsed, backticks neutralised,
    hard length cap. Applied to every string that enters an LLM packet."""
    s = _WS.sub(" ", _CTRL.sub(" ", str(value))).strip().replace("`", "'")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def pattern_signature(value: Any) -> str:
    """'AB-1234' -> 'AA-9999', '05/03/2026' -> '99/99/9999'. Shows the SHAPE of
    a value without its content, so no word from a hostile cell survives."""
    out = []
    for ch in str(value)[: MAX_PATTERN_LEN * 2]:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha():
            out.append("A" if ch.isupper() else "a")
        elif ch.isspace():
            out.append(" ")
        else:
            out.append(ch)
    return sanitize_text("".join(out), MAX_PATTERN_LEN)


# ── row identity and flags ───────────────────────────────────────────────────
class RowId(NamedTuple):
    """Source coordinate: the row number the user sees in their own file.
    sheet is None for CSV (row = line number). Assigned at load, never renumbered."""
    sheet: Optional[str]
    row: int

    def label(self) -> str:
        return f"{self.sheet}!{self.row}" if self.sheet else f"line {self.row}"


@dataclass(frozen=True)
class Flag:
    row_id: RowId
    column: Optional[str]            # header name; None for row-level findings
    column_letter: Optional[str]     # e.g. "D" in a spreadsheet
    reason_code: ReasonCode
    detector: str                    # name@version
    severity: Severity
    proposal_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.row_id, RowId) or self.row_id.row < 1:
            raise ValueError("Flag.row_id must be a RowId with row >= 1 (source coordinates start at 1)")
        if not self.detector:
            raise ValueError("Flag.detector is required")


# ── ops ──────────────────────────────────────────────────────────────────────
_CAST_TARGETS = {"number", "integer", "string", "boolean"}
_IMPUTE = {"median", "mean", "mode", "constant"}


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_op(name: OpName, params: dict) -> dict:
    """Returns a normalised copy of params or raises ValueError. Unknown
    params are rejected, not ignored."""
    if not isinstance(params, dict):
        raise ValueError("op params must be an object")
    p = dict(params)

    def only(*allowed: str) -> None:
        extra = set(p) - set(allowed)
        if extra:
            raise ValueError(f"op '{name.value}': unknown params {sorted(extra)}")

    if name is OpName.CAST:
        only("to")
        if p.get("to") not in _CAST_TARGETS:
            raise ValueError(f"cast.to must be one of {sorted(_CAST_TARGETS)}")
    elif name is OpName.PARSE_DATE:
        only("dayfirst", "format")
        if not isinstance(p.get("dayfirst"), bool):
            raise ValueError("parse_date.dayfirst must be true or false (never guessed)")
        if "format" in p:
            p["format"] = sanitize_text(p["format"], 40)
    elif name is OpName.MAP_VALUES:
        only("mapping")
        m = p.get("mapping")
        if not isinstance(m, dict) or not m or len(m) > MAX_MAP_ENTRIES:
            raise ValueError(f"map_values.mapping must have 1..{MAX_MAP_ENTRIES} entries")
        p["mapping"] = {sanitize_text(k, 64): sanitize_text(v, 64) for k, v in m.items()}
    elif name is OpName.IMPUTE:
        only("strategy", "value")
        if p.get("strategy") not in _IMPUTE:
            raise ValueError(f"impute.strategy must be one of {sorted(_IMPUTE)}")
        if (p["strategy"] == "constant") != ("value" in p):
            raise ValueError("impute.value is required for, and only for, strategy 'constant'")
    elif name is OpName.DEDUPE:
        only("keep")
        if p.get("keep") not in {"first", "last"}:
            raise ValueError("dedupe.keep must be 'first' or 'last'")
    elif name is OpName.CLIP:
        only("lower", "upper")
        lo, hi = p.get("lower"), p.get("upper")
        if lo is None and hi is None:
            raise ValueError("clip needs at least one bound")
        for b in (lo, hi):
            if b is not None and not _is_num(b):
                raise ValueError("clip bounds must be finite numbers")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("clip.lower must not exceed clip.upper")
    else:  # trim, quarantine, flag, blank take no parameters
        only()
    return p


@dataclass(frozen=True)
class Op:
    name: OpName
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", validate_op(self.name, self.params))


# ── declared effect and its verification ─────────────────────────────────────
@dataclass(frozen=True)
class DeclaredEffect:
    """What an op promises. Everything defaults to 0: an op that says nothing
    declares that it changes nothing, so an undeclared side effect fails."""
    rows_removed_max: int = 0
    rows_quarantined_max: int = 0
    cells_changed_max: int = 0
    nulls_added_max: int = 0
    sums_preserved: tuple = ()      # column names whose sum must not change

    def __post_init__(self) -> None:
        for f in ("rows_removed_max", "rows_quarantined_max", "cells_changed_max", "nulls_added_max"):
            v = getattr(self, f)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"DeclaredEffect.{f} must be a non-negative integer")
        object.__setattr__(self, "sums_preserved", tuple(str(c) for c in self.sums_preserved))


def check_effects(declared: DeclaredEffect, actual: dict) -> list[str]:
    """Violations of the promise, as human-readable strings (empty = verified).
    `actual` keys: rows_removed, rows_quarantined, cells_changed, nulls_added
    (missing = 0) and sums: {column: (before, after)}."""
    out = []
    for key, limit in (("rows_removed", declared.rows_removed_max),
                       ("rows_quarantined", declared.rows_quarantined_max),
                       ("cells_changed", declared.cells_changed_max),
                       ("nulls_added", declared.nulls_added_max)):
        got = int(actual.get(key, 0))
        if got > limit:
            out.append(f"{key}: {got} but the op declared at most {limit}")
    sums = actual.get("sums", {})
    for col in declared.sums_preserved:
        if col not in sums:
            out.append(f"sum of '{col}' was declared preserved but could not be verified")
            continue
        before, after = sums[col]
        if not math.isclose(before, after, rel_tol=1e-9, abs_tol=1e-9):
            out.append(f"sum of '{col}' changed ({before} -> {after}) but was declared preserved")
    return out


# ── evidence ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LlmVisible:
    """Aggregates only. This is the part of a proposal's evidence a prompt may use."""
    counts: dict = field(default_factory=dict)      # {"rows_affected": 412}
    shares: dict = field(default_factory=dict)      # {"of_column": 0.034}
    patterns: tuple = ()                            # ((signature, share), ...)

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", {sanitize_text(k, 40): int(v) for k, v in self.counts.items()})
        object.__setattr__(self, "shares", {sanitize_text(k, 40): float(v) for k, v in self.shares.items()})
        object.__setattr__(self, "patterns", tuple(
            (pattern_signature(s), float(sh)) for s, sh in list(self.patterns)[:5]))


@dataclass(frozen=True)
class UserOnly:
    """Raw examples for the review screen. Never serialised into a prompt."""
    examples: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "examples", tuple(
            sanitize_text(e, MAX_EXAMPLE_LEN) for e in list(self.examples)[:MAX_EXAMPLES]))


@dataclass(frozen=True)
class Evidence:
    llm_visible: LlmVisible = field(default_factory=LlmVisible)
    user_only: UserOnly = field(default_factory=UserOnly)


_COLUMN_TYPES = {"integer", "number", "string", "boolean", "date", "datetime", "mixed", "empty"}


@dataclass(frozen=True)
class ColumnEvidence:
    name: str
    position: int
    inferred_type: str
    null_rate: float
    cardinality: int
    numeric: Optional[dict] = None          # {"min","max","quantiles":[5 numbers]}
    patterns: tuple = ()                    # ((value-shape, share), ...)
    top_k: tuple = ()                       # ((value, count), ...) — gated below
    pii_ok: bool = False                    # the PII layer cleared this column's values
    detector_hits: tuple = ()               # ((reason_code, count), ...)

    def __post_init__(self) -> None:
        s = lambda k, v: object.__setattr__(self, k, v)
        s("name", sanitize_text(self.name, MAX_NAME_LEN))
        if self.inferred_type not in _COLUMN_TYPES:
            raise ValueError(f"inferred_type must be one of {sorted(_COLUMN_TYPES)}")
        if not (0.0 <= self.null_rate <= 1.0):
            raise ValueError("null_rate must be within [0, 1]")
        if self.numeric is not None:
            vals = [self.numeric.get("min"), self.numeric.get("max"), *self.numeric.get("quantiles", [])]
            if not all(_is_num(v) for v in vals):
                raise ValueError("numeric stats must be finite numbers")
        s("patterns", tuple((pattern_signature(p), round(float(sh), 4)) for p, sh in list(self.patterns)[:5]))
        # Top-k values are the riskiest field: low cardinality AND PII-cleared, or nothing.
        if self.pii_ok and 0 < self.cardinality <= TOPK_MAX_CARDINALITY:
            s("top_k", tuple((sanitize_text(v, MAX_VALUE_LEN), int(c)) for v, c in list(self.top_k)[:TOPK_MAX_VALUES]))
        else:
            s("top_k", ())
        s("detector_hits", tuple((ReasonCode(c).value, int(n)) for c, n in self.detector_hits))

    def hits(self) -> int:
        return sum(n for _, n in self.detector_hits)

    def to_dict(self) -> dict:
        d = {"name": self.name, "pos": self.position, "type": self.inferred_type,
             "null_rate": round(self.null_rate, 4), "distinct": self.cardinality}
        if self.numeric is not None:
            d["numeric"] = self.numeric
        if self.patterns:
            d["patterns"] = [list(p) for p in self.patterns]
        if self.top_k:
            d["top_k"] = [list(t) for t in self.top_k]
        if self.detector_hits:
            d["detectors"] = [list(h) for h in self.detector_hits]
        return d


def estimate_tokens(obj: Any) -> int:
    """Deliberately pessimistic (JSON is punctuation-heavy): ~1 token per 3 chars."""
    return len(json.dumps(obj, separators=(",", ":"), ensure_ascii=False)) // 3 + 1


def build_packet(n_rows: int, n_cols: int, columns: list, layout: Optional[dict] = None,
                 dataset_detectors: tuple = (), budget_tokens: int = PACKET_TOKEN_BUDGET) -> dict:
    """The only LLM-facing view of a dataset. Takes ColumnEvidence objects only,
    so row values cannot be passed in by accident. Always within budget_tokens
    (unless a single column alone exceeds it, which the caps make impossible)."""
    notes: list[str] = []
    ranked = sorted(columns, key=lambda c: (-c.hits(), c.position))
    omitted = max(0, len(ranked) - MAX_PACKET_COLUMNS)
    kept = list(ranked[:MAX_PACKET_COLUMNS])
    if omitted:
        notes.append(f"{omitted} lower-priority column(s) omitted (column cap {MAX_PACKET_COLUMNS})")

    safe_layout = {sanitize_text(k, 40): (v if isinstance(v, (bool, int, float)) else sanitize_text(v, 60))
                   for k, v in (layout or {}).items()}

    def render() -> dict:
        return {"dataset": {"rows": int(n_rows), "columns": int(n_cols), "layout": safe_layout,
                            "detectors": [[ReasonCode(c).value, int(n)] for c, n in dataset_detectors]},
                "columns": [c.to_dict() for c in kept], "omitted_columns": omitted, "notes": notes}

    def shrink(fn, note: str) -> bool:
        before = json.dumps(render(), sort_keys=True)
        fn()
        if json.dumps(render(), sort_keys=True) != before:
            notes.append(note)
            return True
        return False

    def drop_topk(only_quiet: bool) -> None:
        for i, c in enumerate(kept):
            if c.top_k and (c.hits() == 0 or not only_quiet):
                kept[i] = replace(c, top_k=())

    def trim_patterns() -> None:
        for i, c in enumerate(kept):
            if c.hits() == 0 and len(c.patterns) > 2:
                kept[i] = replace(c, patterns=c.patterns[:2])

    packet = render()
    for fn, note in ((lambda: drop_topk(True), "top-k values dropped for columns with no findings"),
                     (trim_patterns, "pattern lists shortened for columns with no findings"),
                     (lambda: drop_topk(False), "top-k values dropped everywhere")):
        if estimate_tokens(packet) <= budget_tokens:
            break
        shrink(fn, note)
        packet = render()
    while estimate_tokens(packet) > budget_tokens and len(kept) > 1:
        kept.pop()
        omitted += 1
        if not any(n.startswith("columns dropped") for n in notes):
            notes.append("columns dropped to fit the token budget")
        packet = render()
    packet["est_tokens"] = estimate_tokens(packet)
    return packet


# ── proposals ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Scope:
    kind: ScopeKind
    columns: tuple = ()
    sheet: Optional[str] = None
    row_count: int = 0
    flags_ref: Optional[str] = None   # a reference into the flags table; never row data

    def __post_init__(self) -> None:
        if self.kind is ScopeKind.CELLS and not self.columns:
            raise ValueError("a cell-scope proposal must name at least one column")
        if not isinstance(self.row_count, int) or self.row_count < 0:
            raise ValueError("Scope.row_count must be a non-negative integer")
        object.__setattr__(self, "columns", tuple(sanitize_text(c, MAX_NAME_LEN) for c in self.columns))


@dataclass(frozen=True)
class Confidence:
    value: float
    basis: Basis

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("confidence must be within [0, 1]")


_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def make_proposal_id(reason_code: ReasonCode, scope: Scope, op: Op) -> str:
    """Stable across reruns, so a user's decisions don't churn."""
    canon = json.dumps({"r": reason_code.value, "k": scope.kind.value, "s": scope.sheet,
                        "c": sorted(scope.columns), "o": op.name.value, "p": op.params},
                       sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Proposal:
    id: str
    tier: Tier
    reason_code: ReasonCode
    scope: Scope
    evidence: Evidence
    op: Op
    declared_effect: DeclaredEffect
    confidence: Confidence
    reversible: bool
    default_action: Choice
    source: str                               # "detector:<name>" | "llm_contract" | "llm_planner"
    user_choice: Optional[Choice] = None      # None until the user decides

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.id):
            raise ValueError("Proposal.id must be 12 lowercase hex chars (use Proposal.create)")
        if not self.source:
            raise ValueError("Proposal.source is required")
        allowed = _SCOPE_CHOICES[self.scope.kind]
        for label, c in (("default_action", self.default_action), ("user_choice", self.user_choice)):
            if c is not None and c not in allowed:
                raise ValueError(f"{label} '{c.value}' is not valid for {self.scope.kind.value}-scope proposals "
                                 f"(allowed: {sorted(x.value for x in allowed)})")
        # Defaults are tied to the tier: this is what keeps the review screen from rubber-stamping.
        expected = {Tier.SAFE: Choice.APPLY, Tier.REVIEW: Choice.KEEP,
                    Tier.HARD: Choice.QUARANTINE if self.scope.kind is ScopeKind.ROWS else Choice.BLANK}[self.tier]
        if self.default_action is not expected:
            raise ValueError(f"a {self.tier.value}-tier {self.scope.kind.value}-scope proposal must default to "
                             f"'{expected.value}', not '{self.default_action.value}'")
        # Only deterministic evidence may be pre-selected.
        if self.tier in (Tier.SAFE, Tier.HARD) and self.confidence.basis is not Basis.DETERMINISTIC:
            raise ValueError(f"the {self.tier.value} tier requires deterministic evidence, "
                             f"not '{self.confidence.basis.value}'")
        if not self.reversible and self.default_action is not Choice.KEEP:
            raise ValueError("an irreversible op can never be pre-selected")

    @classmethod
    def create(cls, *, tier: Tier, reason_code: ReasonCode, scope: Scope, evidence: Evidence, op: Op,
               declared_effect: DeclaredEffect, confidence: Confidence, source: str,
               reversible: bool = True, default_action: Optional[Choice] = None) -> "Proposal":
        if default_action is None:
            default_action = {Tier.SAFE: Choice.APPLY, Tier.REVIEW: Choice.KEEP,
                              Tier.HARD: Choice.QUARANTINE if scope.kind is ScopeKind.ROWS else Choice.BLANK}[tier]
        return cls(id=make_proposal_id(reason_code, scope, op), tier=tier, reason_code=reason_code, scope=scope,
                   evidence=evidence, op=op, declared_effect=declared_effect, confidence=confidence,
                   reversible=reversible, default_action=default_action, source=source)

    def decide(self, choice: Choice) -> "Proposal":
        if choice not in _SCOPE_CHOICES[self.scope.kind]:
            raise ValueError(f"'{choice.value}' is not a valid choice for {self.scope.kind.value}-scope proposals")
        return replace(self, user_choice=choice)

    def effective_choice(self) -> Choice:
        """What the review screen shows as selected."""
        return self.user_choice or self.default_action

    def llm_view(self) -> dict:
        """The proposal as a prompt may see it: no user_only examples."""
        return {"reason": self.reason_code.value, "tier": self.tier.value, "op": self.op.name.value,
                "columns": list(self.scope.columns), "rows": self.scope.row_count,
                "counts": self.evidence.llm_visible.counts, "shares": self.evidence.llm_visible.shares,
                "patterns": [list(p) for p in self.evidence.llm_visible.patterns]}

    # serialisation (for the review UI and for exportable contracts)
    def to_dict(self) -> dict:
        return _plain(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Proposal":
        """Strict: unknown keys are errors. Used for untrusted input, so it
        constructs through the same validation as everything else."""
        top = {"id", "tier", "reason_code", "scope", "evidence", "op", "declared_effect", "confidence",
               "reversible", "default_action", "source", "user_choice"}
        _no_extra(d, top, "proposal")
        sc = d["scope"]; _no_extra(sc, {"kind", "columns", "sheet", "row_count", "flags_ref"}, "scope")
        ev = d.get("evidence", {}); _no_extra(ev, {"llm_visible", "user_only"}, "evidence")
        lv = ev.get("llm_visible", {}); _no_extra(lv, {"counts", "shares", "patterns"}, "llm_visible")
        uo = ev.get("user_only", {}); _no_extra(uo, {"examples"}, "user_only")
        op = d["op"]; _no_extra(op, {"name", "params"}, "op")
        de = d.get("declared_effect", {})
        _no_extra(de, {"rows_removed_max", "rows_quarantined_max", "cells_changed_max", "nulls_added_max", "sums_preserved"}, "declared_effect")
        cf = d["confidence"]; _no_extra(cf, {"value", "basis"}, "confidence")
        uc = d.get("user_choice")
        return cls(
            id=d["id"], tier=Tier(d["tier"]), reason_code=ReasonCode(d["reason_code"]),
            scope=Scope(kind=ScopeKind(sc["kind"]), columns=tuple(sc.get("columns", ())), sheet=sc.get("sheet"),
                        row_count=sc.get("row_count", 0), flags_ref=sc.get("flags_ref")),
            evidence=Evidence(LlmVisible(lv.get("counts", {}), lv.get("shares", {}), tuple(map(tuple, lv.get("patterns", ())))),
                              UserOnly(tuple(uo.get("examples", ())))),
            op=Op(OpName(op["name"]), op.get("params", {})),
            declared_effect=DeclaredEffect(**{**de, "sums_preserved": tuple(de.get("sums_preserved", ()))}),
            confidence=Confidence(cf["value"], Basis(cf["basis"])),
            reversible=bool(d["reversible"]), default_action=Choice(d["default_action"]), source=d["source"],
            user_choice=Choice(uc) if uc else None)


def _no_extra(d: dict, allowed: set, where: str) -> None:
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be an object")
    extra = set(d) - allowed
    if extra:
        raise ValueError(f"{where}: unknown field(s) {sorted(extra)}")


def _plain(o: Any) -> Any:
    if isinstance(o, Enum):
        return o.value
    if hasattr(o, "__dataclass_fields__"):
        return {k: _plain(getattr(o, k)) for k in o.__dataclass_fields__}
    if isinstance(o, (tuple, list)):
        return [_plain(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    return o


# ── guard rails around contracts and size ────────────────────────────────────
def statistical_checks_enabled(n_rows: int) -> bool:
    """Below SMALL_TABLE_ROWS, outlier / distribution tests are noise: detectors
    skip them and rely on contract ranges and rule checks. (Statistical findings
    can never be pre-selected anyway: the tier rules in Proposal allow them only
    in the review tier, which defaults to keep.)"""
    return n_rows >= SMALL_TABLE_ROWS


def rule_trips_breaker(flagged_rows: int, total_rows: int, max_share: Optional[float] = None) -> bool:
    """A contract rule is a hypothesis. One that flags more than max_share of
    all rows is far more likely wrong than the data: stop and ask the user."""
    limit = CONTRACT_RULE_MAX_FLAG_SHARE if max_share is None else max_share
    return total_rows > 0 and flagged_rows / total_rows > limit


@dataclass(frozen=True)
class CapDecision:
    ok: bool
    message: str = ""


def check_size_cap(n_rows: int, n_cols: int, file_bytes: int = 0, filename: str = "") -> CapDecision:
    """Cleaning is implemented once, in pandas, so it only runs below a size cap.
    When it can't, the user must be told so plainly. Env (read at call time):
    CLEAN_APPLY_MAX_CELLS (rows x columns; unset/0 = unlimited) and
    CLEAN_APPLY_MAX_XLSX_MB (.xlsx expands far more in memory than CSV)."""
    cells = n_rows * n_cols
    max_cells = int(_env_float("CLEAN_APPLY_MAX_CELLS", 0))
    max_xlsx = _env_float("CLEAN_APPLY_MAX_XLSX_MB", 0.0)
    if max_cells and cells > max_cells:
        return CapDecision(False, (
            f"This file has {n_rows:,} rows x {n_cols:,} columns ({cells:,} cells), above the "
            f"{max_cells:,}-cell limit for automatic cleaning on this server. You can still ask questions "
            f"about it, but it has not been cleaned or checked for suspicious rows. "
            f"Splitting the file or removing unused columns would bring it under the limit."))
    if max_xlsx and filename.lower().endswith((".xlsx", ".xlsm")) and file_bytes > max_xlsx * 1024 * 1024:
        return CapDecision(False, (
            f"This Excel file is {file_bytes / 1024 / 1024:.1f} MB, above the {max_xlsx:g} MB limit for "
            f"automatic cleaning on this server (Excel files take far more memory than CSV once loaded). "
            f"Saving it as CSV, or splitting it, would bring it under the limit. "
            f"You can still ask questions about it, but it has not been cleaned or checked."))
    return CapDecision(True)
