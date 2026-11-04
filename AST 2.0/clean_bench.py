"""
clean_bench.py — ground-truth benchmark for the data-cleaning pipeline.

Builds messy CSV files from clean synthetic tables by injecting KNOWN defects,
and records exactly what was injected where. Standard library only, and it
imports nothing from this project: the injectors share no code, regexes or
constants with any detector (a test enforces the import rule).

GROUND TRUTH IS KEPT IN LOGICAL COORDINATES, THEN RENDERED
    A case is a clean typed table + injected defects. The render layer lays it
    out (title lines, header offset, blank rows, totals, footers, ragged rows)
    and only then resolves every label to its SOURCE coordinate: the physical
    line number in data.csv and the 1-based column index. Those are the same
    coordinates the pipeline's RowId uses, so "line 1,043" in a label is
    "line 1,043" in the user's file. An .xlsx writer can be added later as a
    second render layer over the same logical truth.

FILES PER CASE (build_case returns them as bytes; build writes them to disk)
    data.csv     the dirty file, as a user would upload it
    truth.csv    the clean value of every ORIGINAL data row, keyed by its line
                 in data.csv (canonical text: ISO dates, 2-dp money, true/false)
    labels.json  {"meta": {...}, "labels": [...]} — one label per injected
                 defect, per decoy, per column-level condition
    customers.csv (orders only) the companion table that orders.customer_id
                 refers to; orphan keys are defined against it

LABEL (every field always present)
    id, kind ("defect" | "decoy"), defect, scope ("cell" | "row" | "column" |
    "structure" | "file"), line (source line, null for column/file scope),
    cells [{column, col_index, shown, clean}], expected_action, accepted_actions,
    accepted_fixes [{column: canonical value | null}, ...] (any ONE matches),
    composition (compound defects: transforms in the order applied, inner first),
    related_line, extra, note.
    expected_action is one of fix | ask | quarantine | flag | none ("none" =
    must NOT be flagged: decoys). accepted_actions lists every response that is
    also correct, because many defects have more than one valid handling.

DECOYS (legitimate data that merely looks wrong; they drive the false-positive rate)
    genuine heavy-tail values, refunds with a negative amount, real nulls in
    optional columns, rare-but-valid categories, zero stock, and two DISTINCT
    records that share every attribute except their key. Every case, including
    the 0%-defect controls, carries decoys.

DEV vs FROZEN
    The two splits differ in more than their random seeds: they use disjoint
    sentinel strings, date formats, number-in-text styles, locales, spelling-
    variant styles, boolean encodings, unit pairs and whitespace characters,
    and different table sizes. Tuning on dev therefore cannot memorise the
    frozen vocabulary. The manifest records a SHA-256 of each split; it is a
    tripwire, not a lock (the generator is committed, so anyone can regenerate
    the frozen data).

    python clean_bench.py build  [--profile full|mini|coverage] [--out DIR]
    python clean_bench.py verify [--profile full]      # regenerate, compare to MANIFEST.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import random
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Callable, Optional

GENERATOR_VERSION = "1"
D = Decimal
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "eval_data", "clean_bench", "generated")
DEFAULT_MANIFEST = os.path.join(HERE, "eval_data", "clean_bench", "MANIFEST.json")


# ── deterministic randomness (only primitives whose output is stable across Python 3.x) ──
class Rng:
    def __init__(self, seed: str):
        self._r = random.Random(seed)

    def randint(self, a: int, b: int) -> int:
        return self._r.randrange(a, b + 1)

    def unit(self) -> float:
        return self._r.random()

    def chance(self, p: float) -> bool:
        return self._r.random() < p

    def pick(self, seq):
        return seq[self._r.randrange(len(seq))]

    def weighted(self, pairs):
        total = sum(w for _, w in pairs)
        x = self._r.random() * total
        for item, w in pairs:
            x -= w
            if x < 0:
                return item
        return pairs[-1][0]

    def shuffled(self, seq):
        a = list(seq)
        for i in range(len(a) - 1, 0, -1):
            j = self._r.randrange(i + 1)
            a[i], a[j] = a[j], a[i]
        return a


def money(x) -> Decimal:
    return (x if isinstance(x, Decimal) else D(str(x))).quantize(D("0.01"), ROUND_HALF_UP)


# ── vocabularies: dev and frozen are DISJOINT in every slot ──────────────────
VOCAB = {
    "dev": {
        "sentinels": ["N/A", "n/a", "NA", "-", "--", "null", "NULL", "none", "nan"],
        "numeric_sentinels": [],
        "date_formats": ["%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y", "%m/%d/%Y"],
        "number_text_forms": ["grouped", "symbol"],
        "locales": ["plain", "us", "india"],
        "cat_styles": ["upper", "lower", "dot", "double_space"],
        "bool_encodings": [["Y", "N"], ["yes", "no"], ["1", "0"], ["TRUE", "FALSE"]],
        "unit_pairs": [["kg", "g", 1000], ["m", "cm", 100]],
        "whitespace": [" ", "  ", "\t"],
    },
    "frozen": {
        "sentinels": ["#N/A", "?", "unknown", "tba", "nil", "not available", "(blank)", "\u2014", "XXX"],
        "numeric_sentinels": ["-999", "99999"],
        "date_formats": ["%d.%m.%Y", "%Y/%m/%d", "%d %B %Y", "%b %d, %Y", "%Y%m%d"],
        "number_text_forms": ["suffix_usd", "space_group", "apostrophe"],
        "locales": ["eu", "swiss", "suffix_code"],
        "cat_styles": ["typo_swap", "typo_drop", "hyphenate", "amp_and", "diacritic"],
        "bool_encodings": [["T", "F"], ["on", "off"], ["enabled", "disabled"], ["1.0", "0.0"]],
        "unit_pairs": [["l", "ml", 1000], ["km", "m", 1000]],
        "whitespace": ["\u00a0", " \u00a0", "\u3000"],
    },
}
LOCALES = {  # name -> (primary date format, money style)
    "plain": ("%Y-%m-%d", "plain"), "us": ("%m/%d/%Y", "us"), "india": ("%d/%m/%Y", "india"),
    "eu": ("%d.%m.%Y", "eu"), "swiss": ("%d.%m.%Y", "swiss"), "suffix_code": ("%Y/%m/%d", "suffix_code"),
}
AMBIGUOUS_FORMATS = {"%d/%m/%Y", "%m/%d/%Y"}
UNIT_NAMES = {"kg": "weight", "m": "length", "l": "volume", "km": "distance"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


# ── value formatting (own implementations; nothing shared with the pipeline) ─
def fmt_date(d: dt.date, fmt: str) -> str:
    return (fmt.replace("%Y", f"{d.year:04d}").replace("%B", MONTHS[d.month - 1])
            .replace("%b", MONTHS[d.month - 1][:3]).replace("%d", f"{d.day:02d}").replace("%m", f"{d.month:02d}"))


def _group_indian(n: int) -> str:
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail, parts = s[:-3], s[-3:], []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def _split(v: Decimal):
    a = abs(v)
    whole = int(a)
    return v < 0, a, whole, f"{a - whole:.2f}"[1:]


def fmt_money(v: Decimal, style: str) -> str:
    neg, a, whole, cents = _split(v)
    if style == "plain":
        body = f"{a:.2f}"
    elif style == "us":
        body = "$" + f"{whole:,}" + cents
    elif style == "india":
        body = "\u20b9" + _group_indian(whole) + cents
    elif style == "eu":
        body = f"{whole:,}".replace(",", ".") + "," + cents[1:] + " \u20ac"
    elif style == "swiss":
        body = "CHF " + f"{whole:,}".replace(",", "'") + cents
    elif style == "suffix_code":
        body = f"{whole:,}" + cents + " USD"
    else:
        raise ValueError(style)
    return ("-" if neg else "") + body


def text_money(v: Decimal, form: str) -> str:
    neg, a, whole, cents = _split(v)
    if form == "grouped":
        body = f"{whole:,}" + cents
    elif form == "symbol":
        body = "$" + f"{whole:,}" + cents
    elif form == "suffix_usd":
        body = f"{a:.2f} USD"
    elif form == "space_group":
        body = f"{whole:,}".replace(",", " ") + cents
    elif form == "apostrophe":
        body = f"{whole:,}".replace(",", "'") + cents
    else:
        raise ValueError(form)
    return ("-" if neg else "") + body


def style_variant(rng: Rng, style: str, s: str) -> str:
    def typo_swap() -> str:
        if len(s) < 4:
            return s.upper()
        for _ in range(12):
            i = rng.randint(1, len(s) - 3)
            if s[i] != s[i + 1]:
                return s[:i] + s[i + 1] + s[i] + s[i + 2:]
        return s[:1] + s[2:]

    if style == "upper":
        return s.upper()
    if style == "lower":
        return s.lower()
    if style == "dot":
        return s + "."
    if style == "double_space":
        return s.replace(" ", "  ") if " " in s else s.upper()
    if style == "typo_swap":
        return typo_swap()
    if style == "typo_drop":
        i = rng.randint(1, max(1, len(s) - 2))
        return s[:i] + s[i + 1:]
    if style == "hyphenate":
        return s.replace(" ", "-") if " " in s else typo_swap()
    if style == "amp_and":
        return s.replace("&", "and") if "&" in s else typo_swap()
    if style == "diacritic":
        for i, ch in enumerate(s):
            if ch in "aeiou":
                return s[:i] + {"a": "\u00e1", "e": "\u00e9", "i": "\u00ed", "o": "\u00f3", "u": "\u00fa"}[ch] + s[i + 1:]
        return typo_swap()
    raise ValueError(style)


# ── tables ───────────────────────────────────────────────────────────────────
@dataclass
class Col:
    name: str
    kind: str                      # id idnum text cat date money int pct bool unit
    optional: bool = False
    unit: Optional[list] = None    # [base, alt, factor] for kind "unit"


REGIONS = ["North", "South", "East", "West", "Central"]
PRODUCTS = [("Office Supplies", 20), ("Electronics", 18), ("Home & Garden", 16), ("Sports & Outdoors", 14),
            ("Health & Beauty", 12), ("Toys & Games", 10), ("Books", 10)]
STATUSES = [("Completed", 70), ("Shipped", 22), ("Cancelled", 8)]
NOTES = ["Gift wrap", "Express shipping", "Leave at door", "Fragile", "Call before delivery"]
DISCOUNTS = [D("0.00"), D("0.05"), D("0.10"), D("0.15"), D("0.20"), D("0.25"), D("0.30")]
FIRST = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Meera", "Kabir", "Isha", "Arjun", "Sneha", "Neil", "Tara",
         "Dev", "Riya", "Samir"]
LAST = ["Sharma", "Iyer", "Reddy", "Khan", "Patel", "Nair", "Gupta", "Menon", "Das", "Joshi", "Bose", "Rao", "Singh",
        "Verma", "Kulkarni"]
DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Finance", "Operations", "Support"]
WAREHOUSES = ["Mumbai", "Delhi", "Chennai", "Pune", "Hyderabad"]
PARTS = ["Bolt", "Washer", "Bracket", "Gasket", "Valve", "Sensor", "Cable", "Fuse"]
SIZES = ["S", "M", "L", "XL", "10mm", "20mm"]

# family -> per-family behaviour used by the generic injectors
RULES = {
    "orders": {"key": "order_id", "fk": "customer_id", "late_date": ("order_date", "ship_date"),
               "impossible": ("quantity", "neg"), "outlier": "unit_price", "swap": ("customer", "region"),
               "rare": ("category", "Gift Cards"), "total": "amount", "near": "customer", "heavy": "quantity"},
    "staff": {"key": "emp_id", "fk": "manager_id", "late_date": ("start_date", "end_date"),
              "impossible": ("rating", "high"), "outlier": "salary", "swap": ("name", "department"),
              "rare": ("department", "Legal"), "total": "salary", "near": "name", "heavy": "salary"},
    "inventory": {"key": "sku", "fk": None, "late_date": None,
                  "impossible": ("on_hand", "neg"), "outlier": "unit_cost", "swap": ("item", "warehouse"),
                  "rare": ("warehouse", "Cold Storage"), "total": "unit_cost", "near": "item", "heavy": "on_hand"},
}


def family_columns(family: str, vocab: dict) -> list:
    if family == "orders":
        return [Col("order_id", "id"), Col("order_date", "date"), Col("ship_date", "date"),
                Col("customer_id", "idnum"), Col("customer", "text"), Col("region", "cat"), Col("category", "cat"),
                Col("quantity", "int"), Col("unit_price", "money"), Col("discount", "pct"), Col("amount", "money"),
                Col("status", "cat"), Col("paid", "bool"), Col("notes", "text", optional=True)]
    if family == "staff":
        return [Col("emp_id", "id"), Col("name", "text"), Col("department", "cat"), Col("start_date", "date"),
                Col("end_date", "date", optional=True), Col("salary", "money"), Col("rating", "int"),
                Col("active", "bool"), Col("manager_id", "id", optional=True)]
    p0, p1 = vocab["unit_pairs"]
    return [Col("sku", "id"), Col("item", "text"), Col("warehouse", "cat"),
            Col(UNIT_NAMES[p0[0]], "unit", unit=p0), Col(UNIT_NAMES[p1[0]], "unit", unit=p1), Col("bin", "idnum"),
            Col("on_hand", "int"), Col("reorder_level", "int"), Col("unit_cost", "money"),
            Col("last_counted", "date"), Col("in_stock", "bool")]


def pick_date(rng: Rng, lo: dt.date, hi: dt.date, day_max: Optional[int] = None) -> dt.date:
    d = lo + dt.timedelta(days=rng.randint(0, (hi - lo).days))
    if day_max and d.day > day_max:
        d = d.replace(day=rng.randint(1, day_max))
    return d


def gen_rows(family: str, rng: Rng, n: int, ambig: bool, vocab: dict):
    """Clean typed rows (list of dict) and an optional companion table."""
    dm = 12 if ambig else None
    rows, companion = [], None
    if family == "orders":
        used, custs = set(), []
        for _ in range(max(8, n // 5)):
            while True:
                cid = rng.randint(1, 400000)
                if cid not in used:
                    used.add(cid)
                    break
            custs.append((f"{cid:06d}", f"{rng.pick(FIRST)} {rng.pick(LAST)}", rng.pick(REGIONS)))
        companion = {"name": "customers.csv", "header": ["customer_id", "customer", "region"],
                     "rows": [list(c) for c in custs], "ids": {c[0] for c in custs}}
        for i in range(1, n + 1):
            cid, cname, region = rng.pick(custs)
            if ambig:
                od = pick_date(rng, dt.date(2023, 1, 1), dt.date(2025, 12, 31), 6)
                ship = od + dt.timedelta(days=rng.randint(1, 6))
            else:
                od = pick_date(rng, dt.date(2023, 1, 1), dt.date(2025, 12, 31))
                if i == 1:
                    od = od.replace(day=20)
                ship = od + dt.timedelta(days=rng.randint(1, 14))
            q, price, disc = rng.randint(1, 20), money(5 + rng.unit() ** 2 * 395), rng.pick(DISCOUNTS)
            rows.append({"order_id": f"ORD-{i:06d}", "order_date": od, "ship_date": ship, "customer_id": cid,
                         "customer": cname, "region": region, "category": rng.weighted(PRODUCTS), "quantity": q,
                         "unit_price": price, "discount": disc, "amount": money(D(q) * price * (1 - disc)),
                         "status": rng.weighted(STATUSES), "paid": rng.chance(0.85),
                         "notes": rng.pick(NOTES) if rng.chance(0.3) else None})
    elif family == "staff":
        for i in range(1, n + 1):
            start = pick_date(rng, dt.date(2015, 1, 1), dt.date(2024, 6, 30), dm)
            if i == 1 and not ambig:
                start = start.replace(day=20)
            end = None
            if rng.chance(0.15):
                if ambig:
                    end = dt.date(min(start.year + 1 + rng.randint(0, 2), 2025), rng.randint(1, 12), rng.randint(1, 12))
                else:
                    end = start + dt.timedelta(days=rng.randint(200, 2000))
                    end = end if end <= dt.date(2025, 12, 31) else None
            rows.append({"emp_id": f"E{i:04d}", "name": f"{rng.pick(FIRST)} {rng.pick(LAST)}",
                         "department": rng.pick(DEPARTMENTS), "start_date": start, "end_date": end,
                         "salary": money(D(rng.randint(600, 3600) * 50)), "rating": rng.randint(1, 5),
                         "active": end is None,
                         "manager_id": None if i <= 3 else f"E{rng.randint(1, min(i - 1, max(3, i // 4))):04d}"})
    else:
        p0, p1 = vocab["unit_pairs"]
        for i in range(1, n + 1):
            counted = pick_date(rng, dt.date(2025, 1, 1), dt.date(2025, 12, 31), dm)
            if i == 1 and not ambig:
                counted = counted.replace(day=20)
            rows.append({"sku": f"SKU-{i:05d}", "item": f"{rng.pick(PARTS)} {rng.pick(SIZES)}",
                         "warehouse": rng.pick(WAREHOUSES), UNIT_NAMES[p0[0]]: money(0.05 + rng.unit() ** 2 * 24.95),
                         UNIT_NAMES[p1[0]]: money(0.10 + rng.unit() * 11.9), "bin": f"{rng.randint(1, 60000):05d}",
                         "on_hand": rng.randint(1, 500), "reorder_level": rng.randint(5, 60),
                         "unit_cost": money(2 + rng.unit() ** 2 * 98), "last_counted": counted, "in_stock": True})
    return rows, companion


# ── case machinery ───────────────────────────────────────────────────────────
class R:
    """One physical row. kind: junk header data extra blank total footer.
    `clean` holds typed truth (original data rows only); `shown` is what is written."""
    __slots__ = ("kind", "clean", "shown", "line")

    def __init__(self, kind: str, shown: list, clean: Optional[dict] = None):
        self.kind, self.shown, self.clean, self.line = kind, shown, clean, None


@dataclass
class CaseSpec:
    index: int
    split: str
    family: str
    n_rows: int
    locale: str
    rate: float
    bundle: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)

    @property
    def case_id(self) -> str:
        return f"{self.split}-{self.family}-{self.n_rows}-{self.locale}" + ("-ctrl" if self.rate == 0 else "")


JUNK = [["Data export"], ["Generated 2025-12-31 by ERP"], [], ["Prepared for: Finance"]]
FOOT = ["Source: ERP export v4.2", "Confidential - internal use only", "Rows may be revised after month-end close"]
_SCOPE_ORDER = {"file": 0, "structure": 1, "column": 2, "row": 3, "cell": 4}


class Ctx:
    def __init__(self, spec: CaseSpec):
        self.spec = spec
        self.vocab = VOCAB[spec.split]
        self.rng = Rng(f"cleanbench/{GENERATOR_VERSION}/{spec.split}/{spec.case_id}")
        self.rules = RULES[spec.family]
        self.cols = family_columns(spec.family, self.vocab)
        self.ci = {c.name: i for i, c in enumerate(self.cols)}
        self.ambig = spec.flags.get("ambig")  # "dayfirst" | "monthfirst" | None
        loc_fmt, self.money_style = LOCALES[spec.locale]
        self.date_fmt = {"dayfirst": "%d/%m/%Y", "monthfirst": "%m/%d/%Y"}.get(self.ambig, loc_fmt)
        clean, self.companion = gen_rows(spec.family, self.rng, spec.n_rows, bool(self.ambig), self.vocab)
        self.rows = [R("data", [], c) for c in clean]
        for r in self.rows:
            r.shown = self.render_row(r.clean)
        self.free = list(self.rows)
        self.labels: list = []
        self.pending: list = []
        self._vals: dict = {}
        self.footers: list = []
        self.total_row: Optional[R] = None

    # rendering / canonical text
    def render_cell(self, col: Col, v) -> str:
        if v is None:
            return ""
        k = col.kind
        if k == "date":
            return fmt_date(v, self.date_fmt)
        if k == "money":
            return fmt_money(v, self.money_style)
        if k == "bool":
            return "true" if v else "false"
        if k == "unit":
            return f"{v:.2f} {col.unit[0]}"
        return str(v)

    def render_row(self, clean: dict) -> list:
        return [self.render_cell(c, clean[c.name]) for c in self.cols]

    def rerender(self, r: R, names) -> None:
        for n in names:
            i = self.ci[n]
            r.shown[i] = self.render_cell(self.cols[i], r.clean[n])

    def canon(self, col: Col, v) -> Optional[str]:
        if v is None:
            return None
        if col.kind == "date":
            return v.isoformat()
        if col.kind in ("money", "unit"):
            return f"{v:.2f}"
        if col.kind == "bool":
            return "true" if v else "false"
        return str(v)

    def col_values(self, name: str) -> set:
        if name not in self._vals:
            self._vals[name] = {r.clean[name] for r in self.rows if r.clean}
        return self._vals[name]

    # picking
    def cols_idx(self, kinds, allow_optional: bool = True) -> list:
        return [i for i, c in enumerate(self.cols) if c.kind in kinds and (allow_optional or not c.optional)]

    def take(self, pred: Optional[Callable] = None) -> Optional[R]:
        f = self.free
        for _ in range(40):
            if not f:
                return None
            i = self.rng.randint(0, len(f) - 1)
            r = f[i]
            if pred is None or pred(r):
                f[i] = f[-1]
                f.pop()
                return r
        return None

    def target(self, kinds, allow_optional: bool = True, extra: Optional[Callable] = None):
        cis = self.cols_idx(kinds, allow_optional)
        if not cis:
            return None
        ci = self.rng.pick(cis)
        name = self.cols[ci].name
        r = self.take(lambda r: r.clean[name] is not None and (extra is None or extra(r, name)))
        return (r, ci) if r else None

    # labels
    def cell(self, r: R, ci: int, unfixed: Optional[str] = None) -> dict:
        """`unfixed` is what the cell would read if a pipeline only REFORMATTED it (e.g. converted a
        locale-formatted outlier to a plain number) without addressing the defect; None where reformatting
        and fixing are the same act."""
        col = self.cols[ci]
        return {"column": col.name, "col_index": ci + 1, "shown": r.shown[ci] if ci < len(r.shown) else None,
                "clean": self.canon(col, r.clean[col.name]) if r.clean else None, "unfixed": unfixed}

    def add_label(self, defect, scope, row, cis, expected, actions, fixes=None, composition=(), related=None,
                  kind="defect", extra=None, note="", unfixed=None):
        self.labels.append({
            "kind": kind, "defect": defect, "scope": scope, "_row": row, "_related": related,
            "cells": [self.cell(row, ci, (unfixed or {}).get(ci)) for ci in cis] if row is not None else [],
            "expected_action": expected, "accepted_actions": list(actions), "accepted_fixes": fixes or [],
            "composition": list(composition), "extra": extra or {}, "note": note})

    def add_col_label(self, defect, ci, expected, actions, fixes=None, kind="defect", extra=None):
        self.labels.append({
            "kind": kind, "defect": defect, "scope": "column", "_row": None, "_related": None,
            "cells": [{"column": self.cols[ci].name, "col_index": ci + 1, "shown": None, "clean": None, "unfixed": None}],
            "expected_action": expected, "accepted_actions": list(actions), "accepted_fixes": fixes or [],
            "composition": [], "extra": extra or {}, "note": ""})

    def add_file_label(self, defect, scope, expected, actions, extra=None):
        self.labels.append({"kind": "defect", "defect": defect, "scope": scope, "_row": None, "_related": None,
                            "cells": [], "expected_action": expected, "accepted_actions": list(actions),
                            "accepted_fixes": [], "composition": [], "extra": extra or {}, "note": ""})


# ── injectors: each performs ONE injection and returns False when it cannot ──
def inj_sentinel(c: Ctx) -> bool:
    t = c.target(("text", "cat", "date", "money", "int", "pct", "unit"), allow_optional=False)
    if not t:
        return False
    r, ci = t
    col = c.cols[ci]
    pool = c.vocab["sentinels"] + (c.vocab["numeric_sentinels"] if col.kind in ("money", "int", "pct", "unit") else [])
    r.shown[ci] = c.rng.pick(pool)
    c.add_label("sentinel_null", "cell", r, [ci], "fix", ["fix", "blank"], [{col.name: None}])
    return True


def _pad(c: Ctx, s: str) -> str:
    ws, mode = c.rng.pick(c.vocab["whitespace"]), c.rng.pick(["lead", "trail", "both"])
    return (ws if mode != "trail" else "") + s + (ws if mode != "lead" else "")


def inj_whitespace(c: Ctx) -> bool:
    t = c.target(("text", "cat", "id", "idnum"))
    if not t:
        return False
    r, ci = t
    r.shown[ci] = _pad(c, r.shown[ci])
    name = c.cols[ci].name
    c.add_label("whitespace", "cell", r, [ci], "fix", ["fix"], [{name: c.canon(c.cols[ci], r.clean[name])}],
                composition=["whitespace"])
    return True


def _number_text(c: Ctx, v: Decimal) -> str:
    forms = c.vocab["number_text_forms"]
    form = c.rng.pick(forms)
    txt = text_money(v, form)
    if txt == f"{v:.2f}":
        txt = text_money(v, "symbol" if "symbol" in forms else "suffix_usd")
    return txt


def inj_number_text(c: Ctx) -> bool:
    t = c.target(("money",), False, lambda r, n: r.clean[n] >= 1000) or c.target(("money",), False)
    if not t:
        return False
    r, ci = t
    name = c.cols[ci].name
    r.shown[ci] = _number_text(c, r.clean[name])
    c.add_label("number_as_text", "cell", r, [ci], "fix", ["fix"], [{name: c.canon(c.cols[ci], r.clean[name])}])
    return True


def inj_date_mixed(c: Ctx) -> bool:
    t = c.target(("date",))
    if not t:
        return False
    r, ci = t
    name = c.cols[ci].name
    d = r.clean[name]
    # ISO text equals the clean value, so a cell written that way could not be scored as a defect
    fmts = [f for f in c.vocab["date_formats"] if f not in (c.date_fmt, "%Y-%m-%d") and not (f in AMBIGUOUS_FORMATS and d.day <= 12)]
    if not fmts:
        return False
    fmt = c.rng.pick(fmts)
    r.shown[ci] = fmt_date(d, fmt)
    c.add_label("date_mixed_format", "cell", r, [ci], "fix", ["fix"], [{name: d.isoformat()}], extra={"format": fmt})
    return True


def _variant(c: Ctx, canon: str, name: str) -> Optional[str]:
    existing = c.col_values(name)
    for _ in range(8):
        v = style_variant(c.rng, c.rng.pick(c.vocab["cat_styles"]), canon)
        if v != canon and v not in existing:
            return v
    return None


def inj_cat_variant(c: Ctx) -> bool:
    t = c.target(("cat",))
    if not t:
        return False
    r, ci = t
    name = c.cols[ci].name
    v = _variant(c, r.clean[name], name)
    if v is None:
        return False
    r.shown[ci] = v
    c.add_label("category_variant", "cell", r, [ci], "fix", ["fix"], [{name: r.clean[name]}])
    return True


def inj_percent_scale(c: Ctx) -> bool:
    t = c.target(("pct",), False, lambda r, n: r.clean[n] >= D("0.05"))
    if not t:
        return False
    r, ci = t
    name = c.cols[ci].name
    r.shown[ci] = str(int(r.clean[name] * 100))
    c.add_label("percent_scale", "cell", r, [ci], "fix", ["fix", "ask"], [{name: str(r.clean[name])}])
    return True


def inj_leading_zero(c: Ctx) -> bool:
    t = c.target(("idnum",), False, lambda r, n: r.clean[n].startswith("0"))
    if not t:
        return False
    r, ci = t
    name = c.cols[ci].name
    r.shown[ci] = r.clean[name].lstrip("0") or "0"
    c.add_label("lost_leading_zero", "cell", r, [ci], "fix", ["fix", "ask"], [{name: r.clean[name]}])
    return True


def inj_future_date(c: Ctx) -> bool:
    t = c.target(("date",))
    if not t:
        return False
    r, ci = t
    name = c.cols[ci].name
    d = r.clean[name]
    fut = dt.date(c.rng.randint(2031, 2099), d.month, min(d.day, 28))
    if c.ambig:
        fut = fut.replace(day=min(fut.day, 12))
    r.shown[ci] = fmt_date(fut, c.date_fmt)
    c.add_label("future_date", "cell", r, [ci], "flag", ["flag", "blank", "ask", "quarantine"], [{name: None}],
                extra={"year": fut.year}, unfixed={ci: fut.isoformat()})
    return True


def inj_impossible(c: Ctx) -> bool:
    name, mode = c.rules["impossible"]
    ci = c.ci[name]
    r = c.take(lambda r: r.clean[name] is not None and (mode != "neg" or r.clean[name] > 0))
    if not r:
        return False
    fixes = [{name: None}]
    if mode == "neg":
        r.shown[ci] = "-" + r.shown[ci]
        fixes.insert(0, {name: str(r.clean[name])})
    else:
        r.shown[ci] = c.rng.pick(["0", "7", "9", "10", "99"])
    c.add_label("impossible_value", "cell", r, [ci], "flag", ["flag", "blank", "quarantine", "ask"], fixes,
                unfixed={ci: "-" + str(r.clean[name]) if mode == "neg" else r.shown[ci]})
    return True


def inj_outlier(c: Ctx) -> bool:
    name = c.rules["outlier"]
    ci = c.ci[name]
    r = c.take(lambda r: r.clean[name] is not None)
    if not r:
        return False
    r.shown[ci] = c.render_cell(c.cols[ci], money(r.clean[name] * 100))
    c.add_label("outlier_magnitude", "cell", r, [ci], "flag", ["flag", "ask", "blank"],
                [{name: c.canon(c.cols[ci], r.clean[name])}, {name: None}], extra={"factor": 100},
                unfixed={ci: f"{money(r.clean[name] * 100):.2f}"})
    return True


def inj_unit(c: Ctx) -> bool:
    cis = c.cols_idx(("unit",))
    if not cis:
        return False
    ci = c.rng.pick(cis)
    col = c.cols[ci]
    r = c.take(lambda r: r.clean[col.name] is not None)
    if not r:
        return False
    base, alt, factor = col.unit
    r.shown[ci] = f"{int(r.clean[col.name] * factor)} {alt}"
    c.add_label("unit_mismatch", "cell", r, [ci], "fix", ["fix", "ask"], [{col.name: c.canon(col, r.clean[col.name])}],
                extra={"base_unit": base, "unit_shown": alt})
    return True


def inj_compound_money(c: Ctx) -> bool:
    t = c.target(("money",), False)
    if not t:
        return False
    r, ci = t
    name = c.cols[ci].name
    r.shown[ci] = _pad(c, _number_text(c, r.clean[name]))
    c.add_label("compound", "cell", r, [ci], "fix", ["fix"], [{name: c.canon(c.cols[ci], r.clean[name])}],
                composition=["number_as_text", "whitespace"])
    return True


def inj_compound_category(c: Ctx) -> bool:
    t = c.target(("cat",))
    if not t:
        return False
    r, ci = t
    name = c.cols[ci].name
    v = _variant(c, r.clean[name], name)
    if v is None:
        return False
    r.shown[ci] = _pad(c, v)
    c.add_label("compound", "cell", r, [ci], "fix", ["fix"], [{name: r.clean[name]}],
                composition=["category_variant", "whitespace"])
    return True


def inj_dup_exact(c: Ctx) -> bool:
    r = c.take()
    if not r:
        return False
    c.pending.append(("dup_exact", r))
    return True


def inj_dup_near(c: Ctx) -> bool:
    r = c.take()
    if not r:
        return False
    c.pending.append(("dup_near", r))
    return True


def inj_end_before_start(c: Ctx) -> bool:
    a, b = c.rules["late_date"]
    ib = c.ci[b]
    r = c.take(lambda r: r.clean[b] is not None)
    if not r:
        return False
    early = r.clean[a] - dt.timedelta(days=c.rng.randint(1, 30))
    r.shown[ib] = fmt_date(early, c.date_fmt)
    c.add_label("end_before_start", "cell", r, [ib], "quarantine", ["quarantine", "blank", "ask", "flag"], [{b: None}],
                extra={"start_column": a}, unfixed={ib: early.isoformat()})
    return True


def inj_orphan_fk(c: Ctx) -> bool:
    fk = c.rules["fk"]
    ci = c.ci[fk]
    r = c.take(lambda r: r.clean[fk] is not None)
    if not r:
        return False
    orders = c.spec.family == "orders"
    r.shown[ci] = f"{c.rng.randint(900000, 999999):06d}" if orders else f"X{c.rng.randint(1000, 9999)}"
    c.add_label("orphan_fk", "cell", r, [ci], "flag", ["flag", "quarantine", "ask"],
                [{fk: r.clean[fk]}, {fk: None}],
                extra={"references": "customers.csv:customer_id" if orders else "emp_id"})
    return True


def inj_swap(c: Ctx) -> bool:
    a, b = c.rules["swap"]
    ia, ib = c.ci[a], c.ci[b]
    r = c.take()
    if not r:
        return False
    r.shown[ia], r.shown[ib] = r.shown[ib], r.shown[ia]
    c.add_label("swapped_values", "row", r, [ia, ib], "flag", ["flag", "quarantine", "ask", "fix"],
                [{a: r.clean[a], b: r.clean[b]}])
    return True


@dataclass
class Defect:
    name: str
    needs: Callable
    fn: Callable


_PLAIN = lambda c: c.spec.locale == "plain"
DEFECTS = [
    Defect("sentinel_null", lambda c: bool(c.cols_idx(("text", "cat", "date", "money", "int", "pct", "unit"), False)), inj_sentinel),
    Defect("whitespace", lambda c: bool(c.cols_idx(("text", "cat", "id", "idnum"))), inj_whitespace),
    Defect("number_as_text", lambda c: _PLAIN(c) and bool(c.cols_idx(("money",), False)), inj_number_text),
    Defect("date_mixed_format", lambda c: not c.ambig and bool(c.cols_idx(("date",))), inj_date_mixed),
    Defect("category_variant", lambda c: bool(c.cols_idx(("cat",))), inj_cat_variant),
    Defect("percent_scale", lambda c: bool(c.cols_idx(("pct",), False)), inj_percent_scale),
    Defect("lost_leading_zero", lambda c: bool(c.cols_idx(("idnum",), False)), inj_leading_zero),
    Defect("future_date", lambda c: bool(c.cols_idx(("date",))), inj_future_date),
    Defect("impossible_value", lambda c: bool(c.rules["impossible"]), inj_impossible),
    Defect("outlier_magnitude", lambda c: bool(c.rules["outlier"]), inj_outlier),
    Defect("unit_mismatch", lambda c: bool(c.cols_idx(("unit",))), inj_unit),
    Defect("compound_money", lambda c: _PLAIN(c) and bool(c.cols_idx(("money",), False)), inj_compound_money),
    Defect("compound_category", lambda c: bool(c.cols_idx(("cat",))), inj_compound_category),
    Defect("dup_exact", lambda c: True, inj_dup_exact),
    Defect("dup_near", lambda c: True, inj_dup_near),
    Defect("end_before_start", lambda c: bool(c.rules["late_date"]) and not c.ambig, inj_end_before_start),
    Defect("orphan_fk", lambda c: bool(c.rules["fk"]), inj_orphan_fk),
    Defect("swapped_values", lambda c: bool(c.rules["swap"]), inj_swap),
]


def applicable_defects(c: Ctx) -> list:
    return sorted((d for d in DEFECTS if d.needs(c)), key=lambda d: d.name)


def run_defects(c: Ctx) -> None:
    if c.spec.rate <= 0:
        return
    app = applicable_defects(c)
    if not app:
        return
    n = c.spec.n_rows
    k = len(app) if n >= 1000 else min(len(app), max(4, n // 25))
    start = (c.spec.index * 7) % len(app)
    chosen = (app[start:] + app[:start])[:k]
    per = max(1, round(c.spec.rate * n / k))
    per = min(per, max(1, int(0.4 * len(c.free) / k)))
    for d in chosen:
        for _ in range(per):
            if not d.fn(c):
                break


# ── decoys: valid data that merely looks wrong ───────────────────────────────
def place_decoys(c: Ctx) -> None:
    n, rng, rules, fam = c.spec.n_rows, c.rng, c.rules, c.spec.family
    for _ in range(max(1, n // 500)):  # genuine heavy tail
        r = c.take()
        if not r:
            break
        if fam == "orders":
            r.clean["quantity"] = rng.randint(400, 900)
            r.clean["amount"] = money(D(r.clean["quantity"]) * r.clean["unit_price"] * (1 - r.clean["discount"]))
            names = ["quantity", "amount"]
        elif fam == "staff":
            r.clean["salary"] = money(D(rng.randint(9000, 15000) * 100))
            names = ["salary"]
        else:
            r.clean["on_hand"] = rng.randint(5000, 9000)
            names = ["on_hand"]
        c.rerender(r, names)
        c.add_label("heavy_tail_valid", "cell", r, [c.ci[x] for x in names], "none", ["none"], kind="decoy")
    rare_col, rare_val = rules["rare"]
    for _ in range(1 if n < 100 else 2):  # rare but valid category
        r = c.take()
        if not r:
            break
        r.clean[rare_col] = rare_val
        c.rerender(r, [rare_col])
        c.add_label("rare_valid_category", "cell", r, [c.ci[rare_col]], "none", ["none"], kind="decoy")
    for _ in range(max(1, n // 300) if fam == "orders" else 0):  # refund: negative amount is legitimate
        r = c.take()
        if not r:
            break
        r.clean["status"], r.clean["amount"] = "Refunded", -r.clean["amount"]
        c.rerender(r, ["status", "amount"])
        c.add_label("refund_negative_amount", "cell", r, [c.ci["status"], c.ci["amount"]], "none", ["none"], kind="decoy")
    for _ in range(max(1, n // 300) if fam == "inventory" else 0):  # zero stock is not negative stock
        r = c.take()
        if not r:
            break
        r.clean["on_hand"], r.clean["in_stock"] = 0, False
        c.rerender(r, ["on_hand", "in_stock"])
        c.add_label("zero_stock_valid", "cell", r, [c.ci["on_hand"]], "none", ["none"], kind="decoy")
    key = rules["key"]
    for j in range(1, max(1, n // 400) + 1):  # two DISTINCT records that share everything but their key
        src = c.take()
        if not src:
            break
        clone = dict(src.clean)
        clone[key] = {"orders": f"ORD-{n + j:06d}", "staff": f"E{n + j:04d}", "inventory": f"SKU-{n + j:05d}"}[fam]
        row = R("data", c.render_row(clone), clone)
        c.rows.insert(rng.randint(0, len(c.rows)), row)
        c.add_label("same_attributes_distinct_key", "cell", row, [c.ci[key]], "none", ["none"], related=src, kind="decoy")


def column_conditions(c: Ctx) -> None:
    spec = c.spec
    if c.money_style != "plain":
        for ci in c.cols_idx(("money",)):
            c.add_col_label("number_formatted_text", ci, "fix", ["fix"], extra={"locale": spec.locale, "style": c.money_style})
    for ci in c.cols_idx(("date",)):
        if c.ambig:
            c.add_col_label("date_ambiguous", ci, "ask", ["ask"],
                            extra={"truth": c.ambig, "candidates": ["dayfirst", "monthfirst"]})
        elif c.date_fmt != "%Y-%m-%d":
            c.add_col_label("date_locale_format", ci, "fix", ["fix"], extra={"format": c.date_fmt})
    if spec.flags.get("bool_mixed"):
        bis = c.cols_idx(("bool",))
        if bis:
            ci = bis[0]
            name = c.cols[ci].name
            encs = c.rng.shuffled(c.vocab["bool_encodings"])[: c.rng.randint(2, 3)]
            for r in c.rows:
                pair = c.rng.pick(encs)
                r.shown[ci] = pair[0] if r.clean[name] else pair[1]
            c.add_col_label("boolean_mixed_encoding", ci, "fix", ["fix"], extra={"encodings": encs})


# ── structure (render layer) ─────────────────────────────────────────────────
def apply_structure(c: Ctx) -> None:
    b, n, rng = c.spec.bundle, c.spec.n_rows, c.rng
    ncols = len(c.cols)
    if b.get("ragged_rate"):
        for _ in range(max(1, round(b["ragged_rate"] * n))):
            r = c.take()
            if not r:
                break
            if rng.pick(["short", "long"]) == "short" and ncols > 2:
                r.shown = r.shown[: ncols - rng.randint(1, 2)]
            else:
                r.shown = r.shown + ["extra"]
            c.add_label("ragged_row", "row", r, [], "quarantine", ["quarantine", "flag", "ask"],
                        extra={"fields_expected": ncols, "fields_found": len(r.shown)})
    for kind, src in c.pending:
        shown = list(src.shown)
        if kind == "dup_near":
            ci = c.ci[c.rules["near"]]
            shown[ci] = shown[ci].upper() + " "
        row = R("extra", shown)
        c.rows.insert(rng.randint(0, len(c.rows)), row)
        if kind == "dup_exact":
            c.add_label("dup_exact", "row", row, [], "quarantine", ["quarantine", "fix"], related=src)
        else:
            c.add_label("dup_near", "row", row, [], "flag", ["flag", "quarantine", "ask", "fix"], related=src,
                        extra={"changed_column": c.rules["near"]})
    if b.get("blank_rate"):
        for _ in range(max(1, round(b["blank_rate"] * n))):
            row = R("blank", [])
            c.rows.insert(rng.randint(0, len(c.rows)), row)
            c.add_label("blank_row", "row", row, [], "quarantine", ["quarantine", "ignore"])
    if b.get("total_row"):
        tcol = c.rules["total"]
        total = sum((r.clean[tcol] for r in c.rows if r.kind == "data"), D("0"))
        shown = ["TOTAL"] + [""] * (ncols - 1)
        shown[c.ci[tcol]] = fmt_money(total, c.money_style)
        c.total_row = R("total", shown)
        c.labels.append({"kind": "defect", "defect": "total_row", "scope": "row", "_row": c.total_row, "_related": None,
                         "cells": [{"column": tcol, "col_index": c.ci[tcol] + 1, "shown": shown[c.ci[tcol]],
                                    "clean": f"{total:.2f}"}],
                         "expected_action": "quarantine", "accepted_actions": ["quarantine", "ignore"],
                         "accepted_fixes": [], "composition": [], "extra": {"sums": tcol}, "note": ""})
    for t in FOOT[: b.get("footer", 0)]:
        row = R("footer", [t])
        c.footers.append(row)
        c.add_label("footer_note", "row", row, [], "quarantine", ["quarantine", "ignore"])
    if b.get("bom"):
        c.add_file_label("bom", "file", "fix", ["fix"])
    if b.get("trailing_delim"):
        c.add_file_label("trailing_delimiter", "file", "fix", ["fix"])
    if b.get("header_line", 1) > 1:
        c.add_file_label("header_offset", "structure", "fix", ["fix", "ask"],
                         extra={"header_line": b["header_line"], "junk_lines": b["header_line"] - 1})


# ── render + build one case ──────────────────────────────────────────────────
def build_case(spec: CaseSpec):
    c = Ctx(spec)
    place_decoys(c)
    column_conditions(c)
    run_defects(c)
    apply_structure(c)

    b = spec.bundle
    body: list = []
    hl = b.get("header_line", 1)
    body += [R("junk", list(JUNK[j])) for j in range(hl - 1)]
    header = R("header", [col.name for col in c.cols])
    body.append(header)
    body += c.rows
    if c.total_row:
        body.append(c.total_row)
    body += c.footers
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for i, r in enumerate(body, start=1):
        r.line = i
        w.writerow(r.shown + [""] if b.get("trailing_delim") and r.kind in ("header", "data", "extra", "total") else r.shown)
    data_bytes = buf.getvalue().encode("utf-8-sig" if b.get("bom") else "utf-8")

    tb = io.StringIO()
    tw = csv.writer(tb, lineterminator="\n")
    tw.writerow(["__line"] + [col.name for col in c.cols])
    for r in c.rows:
        if r.kind == "data":
            tw.writerow([r.line] + [c.canon(col, r.clean[col.name]) or "" for col in c.cols])

    for ci in c.cols_idx(("date", "money", "text", "id", "int", "cat", "bool", "pct", "idnum", "unit"), True):
        col = c.cols[ci]
        if col.optional:
            k = sum(1 for r in c.rows if r.kind == "data" and r.clean[col.name] is None)
            if k:
                c.add_col_label("optional_null", ci, "none", ["none"], kind="decoy",
                                extra={"null_count": k, "of_rows": sum(1 for r in c.rows if r.kind == "data")})
    for L in c.labels:
        row, rel = L.pop("_row"), L.pop("_related")
        L["line"] = row.line if row is not None else None
        L["related_line"] = rel.line if rel is not None else None
    c.labels.sort(key=lambda L: (L["line"] or 0, _SCOPE_ORDER[L["scope"]], L["defect"],
                                 L["cells"][0]["col_index"] if L["cells"] else 0))
    for i, L in enumerate(c.labels, start=1):
        L["id"] = f"L{i:05d}"
    ordered = ["id", "kind", "defect", "scope", "line", "cells", "expected_action", "accepted_actions",
               "accepted_fixes", "composition", "related_line", "extra", "note"]
    labels = [{k: L[k] for k in ordered} for L in c.labels]

    counts: dict = {}
    for L in labels:
        key = f"{L['kind']}:{L['defect']}"
        counts[key] = counts.get(key, 0) + 1
    meta = {"case_id": spec.case_id, "split": spec.split, "family": spec.family,
            "generator_version": GENERATOR_VERSION, "n_data_rows": sum(1 for r in c.rows if r.kind == "data"),
            "n_extra_rows": sum(1 for r in c.rows if r.kind == "extra"), "locale": spec.locale,
            "defect_rate": spec.rate, "is_control": spec.rate == 0,
            "encoding": "utf-8-sig" if b.get("bom") else "utf-8", "header_line": hl, "bundle": b, "flags": spec.flags,
            "columns": [{"name": col.name, "kind": col.kind, "optional": col.optional,
                         "format": c.date_fmt if col.kind == "date" else (c.money_style if col.kind == "money" else None)}
                        for col in c.cols],
            "companions": [c.companion["name"]] if c.companion else [], "label_counts": counts}
    files = {"data.csv": data_bytes, "truth.csv": tb.getvalue().encode("utf-8"),
             "labels.json": json.dumps({"meta": meta, "labels": labels}, indent=1, ensure_ascii=False).encode("utf-8")}
    if c.companion:
        cb = io.StringIO()
        cw = csv.writer(cb, lineterminator="\n")
        cw.writerow(c.companion["header"])
        cw.writerows(c.companion["rows"])
        files[c.companion["name"]] = cb.getvalue().encode("utf-8")
    return files, {"meta": meta, "labels": labels}


# ── profiles: dev and frozen differ in shape, locale and vocabulary, not just seed ──
B0 = {}
B_BOM = {"bom": True}
B_HF = {"header_line": 4, "footer": 2}
B_BT = {"blank_rate": 0.01, "total_row": True}
B_RT = {"ragged_rate": 0.01, "trailing_delim": True}
B_ALL = {"bom": True, "header_line": 3, "footer": 3, "blank_rate": 0.005, "total_row": True, "ragged_rate": 0.005}
BM = {"bool_mixed": True}
_T = lambda fam, n, loc, rate, bundle=B0, flags=None: (fam, n, loc, rate, bundle, flags or {})

PROFILES = {
    "full": {
        "dev": [_T("orders", 50, "plain", 0.08), _T("orders", 300, "us", 0.08, B_BOM, BM),
                _T("orders", 2000, "india", 0.04, B_HF, {"ambig": "dayfirst"}), _T("orders", 20000, "plain", 0.02, B_BT),
                _T("staff", 50, "us", 0.08, B_RT, {"ambig": "monthfirst"}), _T("staff", 300, "plain", 0.08, B_ALL, BM),
                _T("staff", 2000, "india", 0.04), _T("staff", 20000, "plain", 0.02, B_BOM, BM),
                _T("inventory", 50, "plain", 0.08, B_HF), _T("inventory", 300, "india", 0.08, B_BT, {"ambig": "dayfirst", "bool_mixed": True}),
                _T("inventory", 2000, "us", 0.04, B_ALL), _T("inventory", 20000, "plain", 0.02),
                _T("orders", 1000, "plain", 0), _T("staff", 1000, "plain", 0), _T("inventory", 1000, "plain", 0)],
        "frozen": [_T("orders", 60, "eu", 0.08, B_ALL), _T("orders", 500, "swiss", 0.08, B_HF, BM),
                   _T("orders", 5000, "suffix_code", 0.04, B_RT), _T("orders", 40000, "eu", 0.02, B_BT),
                   _T("staff", 80, "swiss", 0.08, B_BOM, {"ambig": "dayfirst"}), _T("staff", 700, "suffix_code", 0.08, B_ALL, BM),
                   _T("staff", 6000, "eu", 0.04, B_HF), _T("staff", 30000, "swiss", 0.02, B0, BM),
                   _T("inventory", 70, "eu", 0.08, B_RT), _T("inventory", 600, "suffix_code", 0.08, B_BT, {"ambig": "monthfirst"}),
                   _T("inventory", 4000, "swiss", 0.04, B_BOM, BM), _T("inventory", 30000, "eu", 0.02, B_ALL),
                   # plain-locale cases carry the frozen number-in-text styles (they only apply to unformatted columns)
                   _T("orders", 6000, "plain", 0.08), _T("inventory", 2500, "plain", 0.06, B_BOM),
                   _T("staff", 400, "plain", 0.06, B0, {"ambig": "dayfirst"}), _T("orders", 400, "plain", 0.06, B0, {"ambig": "monthfirst"}),
                   _T("orders", 1500, "plain", 0), _T("staff", 1500, "plain", 0), _T("inventory", 1500, "plain", 0)],
    },
    "mini": {
        "dev": [_T("orders", 120, "us", 0.10, B_ALL, BM), _T("staff", 100, "india", 0.10, B_HF, {"ambig": "dayfirst"}),
                _T("inventory", 100, "plain", 0)],
        "frozen": [_T("orders", 130, "eu", 0.10, B_ALL, BM), _T("staff", 110, "swiss", 0.10, B_RT, {"ambig": "monthfirst"}),
                   _T("inventory", 120, "suffix_code", 0.10, B_BT)],
    },
    "coverage": {
        "dev": [_T("orders", 3000, "plain", 0.04), _T("orders", 3000, "us", 0.04), _T("staff", 3000, "plain", 0.04),
                _T("staff", 3000, "india", 0.04), _T("inventory", 3000, "plain", 0.04), _T("inventory", 3000, "us", 0.04)],
        "frozen": [_T("orders", 3000, "plain", 0.04), _T("orders", 3000, "eu", 0.04), _T("staff", 3000, "plain", 0.04),
                   _T("staff", 3000, "swiss", 0.04), _T("inventory", 3000, "plain", 0.04), _T("inventory", 3000, "suffix_code", 0.04)],
    },
}


def specs_for(profile: str, split: str) -> list:
    return [CaseSpec(i, split, *t) for i, t in enumerate(PROFILES[profile][split])]


# ── manifest ─────────────────────────────────────────────────────────────────
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def vocab_sha() -> str:
    return _sha(json.dumps(VOCAB, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def build(profile: str, out_dir: Optional[str] = None, splits=("dev", "frozen")) -> dict:
    """Builds every case in memory (and on disk when out_dir is given); returns the manifest."""
    manifest = {"generator_version": GENERATOR_VERSION, "profile": profile, "vocab_sha256": vocab_sha(), "splits": {}}
    for split in splits:
        cases, agg, counts = {}, [], {}
        for spec in specs_for(profile, split):
            files, doc = build_case(spec)
            hashes = {name: _sha(b) for name, b in sorted(files.items())}
            cases[spec.case_id] = {"rows": doc["meta"]["n_data_rows"], "files": hashes}
            agg += [f"{spec.case_id}/{n} {h}" for n, h in hashes.items()]
            for k, v in doc["meta"]["label_counts"].items():
                counts[k] = counts.get(k, 0) + v
            if out_dir:
                d = os.path.join(out_dir, split, spec.case_id)
                os.makedirs(d, exist_ok=True)
                for name, b in files.items():
                    with open(os.path.join(d, name), "wb") as f:
                        f.write(b)
        manifest["splits"][split] = {"sha256": _sha("\n".join(sorted(agg)).encode("utf-8")), "cases": cases,
                                     "label_counts": dict(sorted(counts.items()))}
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cleaning-pipeline ground-truth benchmark")
    ap.add_argument("command", choices=["build", "verify"])
    ap.add_argument("--profile", default="full", choices=sorted(PROFILES))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    a = ap.parse_args(argv)
    if a.command == "build":
        m = build(a.profile, a.out)
        path = a.manifest if a.profile == "full" else os.path.join(a.out, f"MANIFEST.{a.profile}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=1, ensure_ascii=False)
        for s, v in m["splits"].items():
            print(f"{s}: {len(v['cases'])} cases, sha256 {v['sha256'][:16]}")
        print(f"manifest written: {path}")
        return 0
    with open(a.manifest, encoding="utf-8") as f:
        committed = json.load(f)
    fresh = build(a.profile)
    bad = [s for s in ("dev", "frozen") if committed["splits"][s]["sha256"] != fresh["splits"][s]["sha256"]]
    if committed["vocab_sha256"] != fresh["vocab_sha256"]:
        bad.append("vocabulary")
    print("MISMATCH in: " + ", ".join(bad) if bad else "verified: regenerated data matches the committed manifest")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
