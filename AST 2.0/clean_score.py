"""
clean_score.py — scores any cleaning pipeline against the clean_bench ground truth.

A pipeline is a function  run(case_id, data: bytes, companions: dict) -> prediction doc
(see clean_adapt.py). It sees only the dirty file and companion tables, never the labels.

    python clean_score.py --adapter legacy              # dev split
    python clean_score.py --adapter legacy --frozen     # frozen split, guarded (see below)

METRICS (per defect type, always with sample counts; there is deliberately no headline number)
    detected     the pipeline did something at the labelled place
    correct      what it did is among the label's accepted responses (and, for a fix, the
                 resulting value is among the accepted fixes)
    wrong        it acted, but wrongly (wrong value / wrong action / autofixed an ambiguity)
    silent       the right rows disappeared but nothing reported it ("dropped_unreported")
    missed       nothing happened
    unsupported  the pipeline cannot take ANY accepted action (e.g. it cannot ask): reported as
                 "action not supported", excluded from the rates, never counted as a miss
    File-level conditions (BOM, trailing delimiter, header line) are judged on the resulting structure:
    "correct" means the file was read correctly, which needs no action, so doing nothing can pass.
    Rates use n - unsupported as the denominator. Labels in cases where the pipeline crashed
    count as misses and are also shown separately.

DAMAGE (denominators stated in the report)
    rows wrongly removed   / rows that are clean or decoys (no defect label on them; the
                             surviving copy of a duplicate pair is not counted)
    cells corrupted        / cells in those rows that were correct before cleaning
    decoy hit rate         / decoy labels; any change, flag, ask or removal is a hit
    control touch rate     / cells (rows) in the 0%-defect control files
    ask-vs-fix             asked / autofixed on the ambiguous columns, plus the OVER-ASK
                           rate: asked where a fix was the right answer
    prediction precision   how many located interventions land on a defect, a decoy, or clean data

FROZEN SAFEGUARDS (a tripwire, not a lock)
    --frozen prints aggregates only, never per-case failures; refuses to run on a dirty git
    tree or when the regenerated frozen data no longer matches the committed manifest; logs
    every run (time, commit, scores digest) to the committed FROZEN_RUNS.jsonl; and prints the
    run count, the last run and any tuning-pattern warnings BEFORE the score.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from typing import Callable, Optional

import clean_bench as cb

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_LOG = os.path.join(HERE, "eval_data", "clean_bench", "FROZEN_RUNS.jsonl")
ACTIONS = {"fix", "ask", "quarantine", "flag", "blank", "keep", "ignore", "dropped_unreported"}
INTERVENE = {"fix", "blank", "flag", "ask", "quarantine", "dropped_unreported"}
REMOVE = {"quarantine", "dropped_unreported"}
ALL_CAPS = {"fix", "ask", "quarantine", "flag", "blank", "keep", "ignore"}
FILE_LEVEL = {"bom", "trailing_delimiter", "header_offset"}
COLUMN_FIX = {"number_formatted_text", "date_locale_format", "boolean_mixed_encoding"}
COLUMN_FIX_THRESHOLD = 0.9


# ── normalisation: how two cell values are compared ──────────────────────────
def norm(kind: str, v) -> str:
    """Numbers by value, dates by ISO day, booleans case-insensitively, everything else
    EXACTLY (whitespace, case and leading zeros are real differences). Formatted text such
    as '$1,234.50' or '12 kg' is NOT parsed: leaving it unconverted must score as unconverted."""
    s = "" if v is None else str(v)
    t = s.strip()
    if kind in ("money", "unit", "pct", "int"):
        if t == "":
            return ""
        try:
            return format(Decimal(t).normalize(), "f")
        except InvalidOperation:
            return s
    if kind == "date":
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})(?:[ T]00:00:00)?", t)
        return m.group(1) if m else s
    if kind == "bool":
        return t.lower() if t.lower() in ("true", "false") else s
    return s


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ── cases ────────────────────────────────────────────────────────────────────
class Case:
    def __init__(self, spec, files: dict, doc: dict):
        self.id, self.spec, self.meta, self.labels = spec.case_id, spec, doc["meta"], doc["labels"]
        self.names = [c["name"] for c in self.meta["columns"]]
        self.kinds = [c["kind"] for c in self.meta["columns"]]
        self.optional = [c["optional"] for c in self.meta["columns"]]
        self.control = self.meta["is_control"]
        self.data = files["data.csv"]
        self.companions = {n: b for n, b in files.items() if n not in ("data.csv", "truth.csv", "labels.json")}
        rd = csv.reader(io.StringIO(self.data.decode("utf-8-sig")))
        self.rows = {}
        for row in rd:
            self.rows[rd.line_num] = row
        tr = csv.reader(io.StringIO(files["truth.csv"].decode("utf-8")))
        next(tr)
        self.truth = {int(r[0]): r[1:] for r in tr}

    def shown(self, ln: int, ci: int) -> str:
        f = self.rows.get(ln, [])
        return f[ci - 1] if ci - 1 < len(f) else ""

    def kind(self, ci: int) -> str:
        return self.kinds[ci - 1]


def load_cases(profile: str, split: str) -> list:
    return [Case(s, *cb.build_case(s)) for s in cb.specs_for(profile, split)]


class PI:
    """A prediction doc, indexed."""

    def __init__(self, doc: dict, case: Case):
        self.ok = doc.get("status") == "ok"
        self.error = doc.get("error", "")
        self.caps = set(doc.get("capabilities", []))
        self.structure = doc.get("structure") or {}
        self.cell, self.row, self.col, self.bycol = {}, {}, {}, {}
        idx = {n: i + 1 for i, n in enumerate(case.names)}
        for p in doc.get("predictions", []):
            if p.get("action") not in ACTIONS:
                continue
            ln, ci = p.get("line"), p.get("col_index") or idx.get(p.get("column"))
            if ln is not None and ci is not None:
                self.cell.setdefault((ln, ci), []).append(p)
                self.bycol.setdefault(ci, []).append(p)
            elif ln is not None:
                self.row.setdefault(ln, []).append(p)
            elif ci is not None:
                self.col.setdefault(ci, []).append(p)
        self.removed = {}
        for ln, ps in self.row.items():
            acts = {p["action"] for p in ps}
            if acts & REMOVE:
                self.removed[ln] = "quarantine" if "quarantine" in acts else "dropped_unreported"

    def final(self, case: Case, ln: int, ci: int) -> str:
        for p in self.cell.get((ln, ci), ()):
            if p["action"] == "fix":
                return "" if p.get("value") is None else str(p["value"])
            if p["action"] == "blank":
                return ""
        return case.shown(ln, ci)


# ── judging one label ────────────────────────────────────────────────────────
def _fix_matches(case, pi, L) -> bool:
    fixes = L["accepted_fixes"]
    if not fixes or not L["cells"]:
        return True
    ln = L["line"]
    got = {c["col_index"]: pi.final(case, ln, c["col_index"]) for c in L["cells"]}
    for fx in fixes:
        if all(norm(case.kind(c["col_index"]), got[c["col_index"]]) == norm(case.kind(c["col_index"]), fx.get(c["column"]))
               for c in L["cells"]):
            return True
    return False


def judge_defect(case, pi, L, colacc):
    """-> (outcome, detail, detected)"""
    d, sc, acc = L["defect"], L["scope"], set(L["accepted_actions"])
    if d == "bom":
        ok = bool(pi.structure.get("columns")) and pi.structure["columns"][0] == case.names[0]
        return ("correct" if ok else "wrong"), "" if ok else "bom leaked into the first column name", ok
    if d == "trailing_delimiter":
        ok = len(pi.structure.get("columns", [])) == len(case.names)
        return ("correct" if ok else "wrong"), "" if ok else "phantom column", ok
    if d == "header_offset":
        ok = pi.structure.get("header_line") == L["extra"]["header_line"]
        return ("correct" if ok else "wrong"), "" if ok else "header not found", ok
    if sc == "column":
        ci = L["cells"][0]["col_index"]
        if d == "date_ambiguous":
            if any(p["action"] == "ask" for p in pi.col.get(ci, [])):
                return "correct", "", True
            if any(p["action"] in ("fix", "blank") for p in pi.bycol.get(ci, [])):
                return "wrong", "autofixed", True
            return "missed", "", False
        if colacc.get(ci, 0.0) >= COLUMN_FIX_THRESHOLD:
            return "correct", "", True
        if pi.bycol.get(ci):
            return "wrong", "column only partly converted", True
        return "missed", "", False

    ln, cis = L["line"], [c["col_index"] for c in L["cells"]]
    if d in ("dup_exact", "dup_near"):
        pair = [ln, L["related_line"]]
        rem = [pi.removed[x] for x in pair if x in pi.removed]
        if len(rem) == 2:
            return "wrong", "both copies removed", True
        if len(rem) == 1:
            return ("correct" if rem[0] == "quarantine" else "silent"), "", True
        for x in pair:
            for p in pi.row.get(x, []):
                if p["action"] in ("flag", "ask"):
                    return ("correct" if p["action"] in acc else "wrong"), "", True
        return "missed", "", False

    preds = [p for ci in cis for p in pi.cell.get((ln, ci), [])] + pi.row.get(ln, [])
    acts = {p["action"] for p in preds}
    if not acts - {"keep"}:
        return "missed", "", False
    # Converting a text-formatted cell to a number without touching the defect is not an attempt at the defect.
    if acts <= {"fix", "keep"} and L["cells"] and all(c.get("unfixed") is not None for c in L["cells"]) and \
            all(norm(case.kind(c["col_index"]), pi.final(case, ln, c["col_index"])) == norm(case.kind(c["col_index"]), c["unfixed"])
                for c in L["cells"]):
        return "missed", "reformatted only", False
    results = []
    if "quarantine" in acts:
        results.append(("correct" if "quarantine" in acc else "wrong", "removed a row that should be handled otherwise"))
    if "dropped_unreported" in acts:
        results.append(("silent" if acc & {"quarantine", "ignore"} else "wrong", "silent removal"))
    if "ignore" in acts:
        results.append(("correct" if "ignore" in acc else "wrong", "ignored"))
    if "ask" in acts:
        results.append(("correct" if "ask" in acc else "wrong", "over-ask" if "fix" in acc else "asked"))
    if "flag" in acts:
        results.append(("correct" if "flag" in acc else "wrong", "flagged"))
    if acts & {"fix", "blank"} and cis:
        finals = [pi.final(case, ln, ci) for ci in cis]
        blank = all(f == "" for f in finals)
        allowed = ("blank" in acc or "fix" in acc) if blank else ("fix" in acc)
        results.append(("correct" if allowed and _fix_matches(case, pi, L) else "wrong", "wrong value"))
    for o, dtl in results:
        if o == "correct":
            return "correct", "", True
    for o, dtl in results:
        if o == "silent":
            return "silent", dtl, True
    return "wrong", results[0][1] if results else "", True


def judge_decoy(case, pi, L, null_cells) -> str:
    if L["defect"] == "optional_null":
        ci = L["cells"][0]["col_index"]
        return "decoy_hit" if any(p["action"] in INTERVENE and (p["line"], ci) in null_cells for p in pi.bycol.get(ci, [])) else "decoy_ok"
    lines = [L["line"]] + ([L["related_line"]] if L.get("related_line") else [])
    cis = [c["col_index"] for c in L["cells"]]
    for ln in lines:
        if ln in pi.removed or any(p["action"] in ("flag", "ask") for p in pi.row.get(ln, [])):
            return "decoy_hit"
    for ci in cis:
        for p in pi.cell.get((L["line"], ci), []):
            if p["action"] in ("flag", "ask"):
                return "decoy_hit"
            if p["action"] in ("fix", "blank"):
                # converting a decoy to its TRUE value (e.g. text-formatted money) is required work, not damage
                k = case.kind(ci)
                if norm(k, pi.final(case, L["line"], ci)) != norm(k, case.truth[L["line"]][ci - 1]):
                    return "decoy_hit"
    return "decoy_ok"


# ── scoring ──────────────────────────────────────────────────────────────────
class Agg:
    def __init__(self):
        self.d = {}      # defect -> counters
        self.dec = {}    # decoy -> [n, hits]
        self.m = {"cases": 0, "cases_failed": 0, "failed": [], "clean_rows": 0, "rows_removed_reported": 0,
                  "rows_removed_silent": 0, "decoy_rows_removed": 0, "good_cells": 0, "cells_corrupted": 0,
                  "cells": 0, "cells_ok_before": 0, "cells_ok_after": 0, "ctl_cells": 0, "ctl_touched": 0,
                  "ctl_rows": 0, "ctl_removed": 0, "preds": 0, "on_defect": 0, "on_decoy": 0, "on_clean": 0,
                  "ask_n": 0, "ask_asked": 0, "ask_autofixed": 0, "ask_unsupported": 0, "fix_n": 0, "over_ask": 0}

    def bump(self, defect: str, key: str, n: int = 1):
        c = self.d.setdefault(defect, {"n": 0, "unsupported": 0, "detected": 0, "correct": 0, "wrong": 0, "silent": 0,
                                       "missed": 0, "case_failed": 0, "detail": {}})
        c[key] += n


def score_case(case: Case, doc: dict, agg: Agg) -> None:
    pi, m = PI(doc, case), agg.m
    m["cases"] += 1
    if not pi.ok:
        m["cases_failed"] += 1
        m["failed"].append((case.id, pi.error))
    lines = sorted(case.truth)
    ncols = len(case.names)
    defect_lines, partner, defect_cells, decoy_cells, decoy_lines, colcond = set(), set(), set(), set(), set(), set()
    for L in case.labels:
        if L["kind"] == "defect" and L["scope"] in ("cell", "row") and L["line"]:
            defect_lines.add(L["line"])
            defect_cells |= {(L["line"], c["col_index"]) for c in L["cells"]}
            if L["related_line"] and L["defect"] in ("dup_exact", "dup_near"):
                partner.add(L["related_line"])
        elif L["kind"] == "defect" and L["scope"] == "column":
            colcond.add(L["cells"][0]["col_index"])
        elif L["kind"] == "decoy" and L["line"]:
            decoy_lines.add(L["line"])
            decoy_cells |= {(L["line"], c["col_index"]) for c in L["cells"]}
    clean_lines = set(lines) - defect_lines - partner

    # cell states
    ok_col, tot_col, null_cells = {}, {}, set()
    for ln in lines:
        removed = ln in pi.removed
        for ci in range(1, ncols + 1):
            k = case.kind(ci)
            tv = norm(k, case.truth[ln][ci - 1])
            if tv == "" and case.optional[ci - 1]:
                null_cells.add((ln, ci))
            before = norm(k, case.shown(ln, ci)) == tv
            after = (not removed) and norm(k, pi.final(case, ln, ci)) == tv
            m["cells"] += 1
            m["cells_ok_before"] += before
            m["cells_ok_after"] += after
            tot_col[ci] = tot_col.get(ci, 0) + 1
            ok_col[ci] = ok_col.get(ci, 0) + after
            if ln in clean_lines and not removed:
                m["good_cells"] += before
                m["cells_corrupted"] += before and not after
    colacc = {ci: ok_col[ci] / tot_col[ci] for ci in tot_col}

    # row damage, over rows that are clean or decoys
    for ln in clean_lines:
        if ln in pi.removed:
            m["rows_removed_reported" if pi.removed[ln] == "quarantine" else "rows_removed_silent"] += 1
            m["decoy_rows_removed"] += ln in decoy_lines
    m["clean_rows"] += len(clean_lines)

    # controls
    if case.control:
        m["ctl_cells"] += len(lines) * ncols
        m["ctl_touched"] += sum(1 for k, ps in pi.cell.items() if k[0] in case.truth and any(p["action"] in INTERVENE for p in ps))
        m["ctl_rows"] += len(lines)
        m["ctl_removed"] += sum(1 for ln in pi.removed if ln in case.truth)

    # prediction precision
    for p in (q for ps in list(pi.cell.values()) + list(pi.row.values()) for q in ps):
        if p["action"] not in INTERVENE:
            continue
        ln, ci = p["line"], p.get("col_index") or {n: i + 1 for i, n in enumerate(case.names)}.get(p.get("column"))
        m["preds"] += 1
        if (ln, ci) in defect_cells or ln in defect_lines or ln in partner or ci in colcond:
            m["on_defect"] += 1
        elif (ln, ci) in decoy_cells or ln in decoy_lines:
            m["on_decoy"] += 1
        else:
            m["on_clean"] += 1

    # labels
    for L in case.labels:
        if L["kind"] == "decoy":
            o = judge_decoy(case, pi, L, null_cells) if pi.ok else "decoy_ok"
            c = agg.dec.setdefault(L["defect"], [0, 0])
            c[0] += 1
            c[1] += o == "decoy_hit"
            continue
        d, acc = L["defect"], set(L["accepted_actions"])
        agg.bump(d, "n")
        if d not in FILE_LEVEL and not (acc & pi.caps):
            agg.bump(d, "unsupported")
            if L["expected_action"] == "ask":
                m["ask_n"] += 1
                m["ask_unsupported"] += 1
            continue
        if L["expected_action"] == "ask":
            m["ask_n"] += 1
        if not pi.ok:
            agg.bump(d, "case_failed")
            agg.bump(d, "missed")
            continue
        outcome, detail, detected = judge_defect(case, pi, L, colacc)
        agg.bump(d, outcome)
        agg.bump(d, "detected", int(detected))
        if detail:
            det = agg.d[d]["detail"]
            det[detail] = det.get(detail, 0) + 1
        if L["expected_action"] == "ask":
            m["ask_asked"] += outcome == "correct"
            m["ask_autofixed"] += detail == "autofixed"
        elif L["expected_action"] == "fix" and "ask" not in acc and L["scope"] != "column" and d not in FILE_LEVEL:
            m["fix_n"] += 1
            m["over_ask"] += any(p["action"] == "ask" for ci in (c["col_index"] for c in L["cells"]) for p in pi.cell.get((L["line"], ci), []))


def score(cases: list, run: Callable) -> dict:
    agg = Agg()
    for case in cases:
        doc = run(case.id, case.data, case.companions)
        score_case(case, doc, agg)
    return finish(agg)


def finish(agg: Agg) -> dict:
    m = agg.m
    rate = lambda a, b: round(a / b, 4) if b else None
    by_defect = {}
    for d, c in sorted(agg.d.items()):
        scored = c["n"] - c["unsupported"]
        lo, hi = wilson(c["correct"], scored)
        by_defect[d] = {**c, "scored": scored, "detected_rate": rate(c["detected"], scored),
                        "correct_rate": rate(c["correct"], scored), "correct_ci95": [round(lo, 3), round(hi, 3)]}
    return {
        "cases": m["cases"], "cases_failed": m["cases_failed"], "failed": m["failed"], "by_defect": by_defect,
        "by_decoy": {k: {"n": v[0], "hits": v[1], "hit_rate": rate(v[1], v[0])} for k, v in sorted(agg.dec.items())},
        "damage": {
            "clean_or_decoy_rows": m["clean_rows"],
            "rows_wrongly_removed_reported": m["rows_removed_reported"],
            "rows_wrongly_removed_silent": m["rows_removed_silent"],
            "rows_wrongly_removed_rate": rate(m["rows_removed_reported"] + m["rows_removed_silent"], m["clean_rows"]),
            "decoy_rows_removed": m["decoy_rows_removed"],
            "cells_correct_before_in_clean_rows": m["good_cells"], "cells_corrupted": m["cells_corrupted"],
            "cells_corrupted_rate": rate(m["cells_corrupted"], m["good_cells"]),
        },
        "end_state": {"cells": m["cells"], "correct_before": rate(m["cells_ok_before"], m["cells"]),
                      "correct_after": rate(m["cells_ok_after"], m["cells"])},
        "controls": {"cells": m["ctl_cells"], "cells_touched_rate": rate(m["ctl_touched"], m["ctl_cells"]),
                     "rows": m["ctl_rows"], "rows_removed_rate": rate(m["ctl_removed"], m["ctl_rows"])},
        "ask_vs_fix": {"ambiguous_columns": m["ask_n"], "unsupported": m["ask_unsupported"], "asked": m["ask_asked"],
                       "autofixed": m["ask_autofixed"], "fix_labels": m["fix_n"], "over_ask": m["over_ask"],
                       "over_ask_rate": rate(m["over_ask"], m["fix_n"])},
        "precision": {"interventions": m["preds"], "on_defect": m["on_defect"], "on_decoy": m["on_decoy"],
                      "on_clean": m["on_clean"], "precision": rate(m["on_defect"], m["preds"])},
    }


# ── reporting ────────────────────────────────────────────────────────────────
def pct(x) -> str:
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def render(r: dict, title: str, show_failures: bool) -> str:
    L = [f"== {title} ==", f"{r['cases']} cases, {r['cases_failed']} crashed (their labels count as misses)", ""]
    L.append(f"{'defect type':<26}{'n':>6}{'unsupp':>8}{'detected':>10}{'correct':>9}{'wrong':>7}{'silent':>8}{'missed':>8}{'in crashed':>12}   correct 95% CI")
    for d, c in r["by_defect"].items():
        L.append(f"{d:<26}{c['n']:>6}{c['unsupported']:>8}{pct(c['detected_rate']):>10}{pct(c['correct_rate']):>9}"
                 f"{c['wrong']:>7}{c['silent']:>8}{c['missed']:>8}{c['case_failed']:>12}   [{c['correct_ci95'][0] * 100:.0f}-{c['correct_ci95'][1] * 100:.0f}%]")
    L += ["", "rates use (n - unsupported) as the denominator; 'unsupp' = the pipeline cannot take any accepted action", "",
          f"{'decoy (must NOT be touched)':<36}{'n':>6}{'hit':>6}{'hit rate':>10}"]
    for d, c in r["by_decoy"].items():
        L.append(f"{d:<36}{c['n']:>6}{c['hits']:>6}{pct(c['hit_rate']):>10}")
    dm, es, ct, av, pr = r["damage"], r["end_state"], r["controls"], r["ask_vs_fix"], r["precision"]
    L += ["", "-- damage --",
          f"rows wrongly removed: {dm['rows_wrongly_removed_reported'] + dm['rows_wrongly_removed_silent']} of {dm['clean_or_decoy_rows']} "
          f"clean-or-decoy rows ({pct(dm['rows_wrongly_removed_rate']).strip()}); {dm['rows_wrongly_removed_silent']} of them silently; "
          f"{dm['decoy_rows_removed']} were decoys",
          f"cells corrupted: {dm['cells_corrupted']} of {dm['cells_correct_before_in_clean_rows']} cells that were correct before cleaning "
          f"({pct(dm['cells_corrupted_rate']).strip()})",
          f"cells matching ground truth: {pct(es['correct_before']).strip()} before -> {pct(es['correct_after']).strip()} after ({es['cells']:,} cells)",
          f"control files (no injected defects): {pct(ct['cells_touched_rate']).strip()} of cells touched, {pct(ct['rows_removed_rate']).strip()} of rows removed",
          f"interventions on defects / decoys / clean data: {pr['on_defect']} / {pr['on_decoy']} / {pr['on_clean']} (precision {pct(pr['precision']).strip()})",
          "", "-- ask vs fix --",
          f"ambiguous date columns: {av['ambiguous_columns']} (unsupported: {av['unsupported']}), asked: {av['asked']}, autofixed: {av['autofixed']}",
          f"over-ask: asked where a fix was right on {av['over_ask']} of {av['fix_labels']} fix labels ({pct(av['over_ask_rate']).strip()})"]
    if show_failures and r["failed"]:
        L += ["", "-- crashed cases --"] + [f"{cid}: {err[:150]}" for cid, err in r["failed"]]
    return "\n".join(L)


# ── reference pipelines (used to test the metrics, and as sanity checks) ──────
def _doc(case_id, preds, structure, caps=ALL_CAPS):
    return {"case_id": case_id, "status": "ok", "error": "", "capabilities": sorted(caps), "structure": structure, "predictions": preds}


def _p(line, ci, action, value=None, name=None):
    return {"line": line, "col_index": ci, "column": name, "action": action, "value": value, "reason": ""}


def oracle_doc(case: Case) -> dict:
    preds, done = [], set()
    for L in case.labels:
        if L["kind"] == "decoy" or L["defect"] in FILE_LEVEL:
            continue
        sc, ln, exp = L["scope"], L["line"], L["expected_action"]
        if sc == "column":
            ci = L["cells"][0]["col_index"]
            if exp == "ask":
                preds.append(_p(None, ci, "ask"))
            continue
        cis = [c["col_index"] for c in L["cells"]]
        if L["defect"] in ("dup_exact", "dup_near", "ragged_row", "blank_row", "footer_note", "total_row") or exp == "quarantine":
            preds.append(_p(ln, None, "quarantine" if L["defect"] != "blank_row" else "ignore"))
        elif exp == "flag":
            preds += [_p(ln, ci, "flag") for ci in cis] if cis else [_p(ln, None, "flag")]
        elif exp == "fix":
            fixes = L["accepted_fixes"]
            fx = next((f for f in fixes if any(v is not None for v in f.values())), fixes[0] if fixes else {})
            for c in L["cells"]:
                v = fx.get(c["column"], c["clean"])
                preds.append(_p(ln, c["col_index"], "blank" if v is None else "fix", v))
        done |= {(ln, ci) for ci in cis}
    for L in case.labels:  # columns that are wholesale text in a locale format
        if L["kind"] == "defect" and L["scope"] == "column" and L["defect"] in COLUMN_FIX:
            ci = L["cells"][0]["col_index"]
            for ln in case.truth:
                if (ln, ci) not in done and norm(case.kind(ci), case.shown(ln, ci)) != norm(case.kind(ci), case.truth[ln][ci - 1]):
                    preds.append(_p(ln, ci, "fix", case.truth[ln][ci - 1]))
    hl = next((L["extra"]["header_line"] for L in case.labels if L["defect"] == "header_offset"), 1)
    return _doc(case.id, preds, {"header_line": hl, "columns": list(case.names)})


def _records(case):
    return [ln for ln in sorted(case.rows) if case.rows[ln]]


def reference_pipeline(name: str, cases: list) -> Callable:
    by_id = {c.id: c for c in cases}

    def run(case_id, data, companions):
        case = by_id[case_id]
        if name == "oracle":
            return oracle_doc(case)
        recs = _records(case)
        structure = {"header_line": 1, "columns": list(case.rows[recs[0]])}
        if name == "null":
            return _doc(case_id, [], structure)
        if name == "flag_all":
            body = recs[1:]
            return _doc(case_id, [_p(ln, ci, "flag") for ln in body for ci in range(1, len(case.names) + 1)]
                        + [_p(ln, None, "flag") for ln in body], structure)
        if name == "corrupter":
            return _doc(case_id, [_p(ln, ci, "fix", case.shown(ln, ci).upper() + "!") for ln in case.truth
                                  for ci in range(1, len(case.names) + 1) if case.shown(ln, ci)], structure)
        if name == "crash":
            return {"case_id": case_id, "status": "error", "error": "boom: the pipeline fell over", "capabilities": sorted(ALL_CAPS),
                    "structure": {}, "predictions": []}
        if name == "dropper":
            return _doc(case_id, [_p(ln, None, "dropped_unreported") for i, ln in enumerate(sorted(case.truth)) if i % 10 == 5], structure)
        raise ValueError(name)

    return run


def make_pipeline(name: str, cases: list) -> Callable:
    if name in ("oracle", "null", "flag_all", "corrupter", "dropper", "crash"):
        return reference_pipeline(name, cases)
    if name == "legacy":
        from clean_adapter_legacy import run
        return run
    raise SystemExit(f"unknown adapter '{name}'")


# ── frozen safeguards ────────────────────────────────────────────────────────
def git_state(cwd: str = HERE):
    try:
        run = lambda *a: subprocess.run(["git", "-C", cwd, *a], capture_output=True, text=True, timeout=30)
        head = run("rev-parse", "HEAD").stdout.strip()
        dirty = [l for l in run("status", "--porcelain").stdout.splitlines() if l.strip()]
        return (head or None), len(dirty)
    except Exception:
        return None, None


def read_log(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def digest_of(result: dict) -> str:
    core = {"d": {k: [v["correct"], v["scored"], v["wrong"], v["silent"]] for k, v in result["by_defect"].items()},
            "x": [result["damage"]["rows_wrongly_removed_rate"], result["damage"]["cells_corrupted_rate"], result["end_state"]["correct_after"]]}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:12]


def frozen_preflight(log_path: str, now: dt.datetime, git: Callable, manifest_ok: Callable) -> tuple:
    """-> (ok, lines to print BEFORE scoring, commit)"""
    commit, dirty = git()
    if commit is None or dirty is None:
        return False, ["REFUSED: cannot read git state, so this run could not be tied to a commit."], None
    if dirty:
        return False, [f"REFUSED: the working tree has {dirty} uncommitted change(s). Commit first so this frozen run maps to a commit."], commit
    if not manifest_ok():
        return False, ["REFUSED: the regenerated frozen data does not match MANIFEST.json (generator or vocabulary changed)."], commit
    runs = read_log(log_path)
    week = [r for r in runs if (now - dt.datetime.fromisoformat(r["ts"])).total_seconds() < 7 * 86400]
    at_commit = [r for r in runs if r["commit"] == commit]
    L = [f"FROZEN RUN #{len(runs) + 1}  |  this week: {len(week) + 1}  |  at this commit: {len(at_commit) + 1}  |  "
         f"last run: {runs[-1]['ts'] + ' @ ' + runs[-1]['commit'][:8] if runs else 'never'}"]
    if runs and (now - dt.datetime.fromisoformat(runs[-1]["ts"])).total_seconds() < 3600:
        L.append("WARNING: the previous frozen run was under an hour ago.")
    if len(week) + 1 >= 5:
        L.append(f"WARNING: {len(week) + 1} frozen runs in 7 days.")
    if len(runs) >= 2 and runs[-1]["commit"] != runs[-2]["commit"] and runs[-1]["digest"] != runs[-2]["digest"]:
        L.append("WARNING: your last two runs were on different commits and scored differently - the signature of tuning on the frozen set.")
    return True, L, commit


def append_log(log_path: str, now: dt.datetime, commit: str, adapter: str, result: dict) -> None:
    rec = {"ts": now.replace(microsecond=0).isoformat(), "commit": commit, "adapter": adapter, "digest": digest_of(result),
           "correct": {k: [v["correct"], v["scored"]] for k, v in result["by_defect"].items()},
           "rows_wrongly_removed_rate": result["damage"]["rows_wrongly_removed_rate"],
           "cells_corrupted_rate": result["damage"]["cells_corrupted_rate"]}
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def main(argv=None, git: Callable = git_state, log_path: str = RUN_LOG, now: Optional[dt.datetime] = None,
         manifest_check: Optional[Callable] = None) -> int:
    ap = argparse.ArgumentParser(description="Score a cleaning pipeline against the clean_bench ground truth")
    ap.add_argument("--adapter", required=True, help="legacy | oracle | null | flag_all | corrupter | dropper")
    ap.add_argument("--frozen", action="store_true", help="score the frozen split (guarded; aggregates only)")
    ap.add_argument("--profile", default="full", choices=sorted(cb.PROFILES))
    ap.add_argument("--json", help="also write the full metrics to this file (dev only)")
    a = ap.parse_args(argv)
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    commit = None
    if a.frozen:
        def manifest_ok():
            try:
                with open(cb.DEFAULT_MANIFEST, encoding="utf-8") as f:
                    committed = json.load(f)
                fresh = cb.build(a.profile, splits=("frozen",))
                return committed["splits"]["frozen"]["sha256"] == fresh["splits"]["frozen"]["sha256"] and committed["vocab_sha256"] == fresh["vocab_sha256"]
            except Exception:
                return False
        ok, lines, commit = frozen_preflight(log_path, now, git, manifest_check or manifest_ok)
        print("\n".join(lines))
        if not ok:
            return 2
        print()
    split = "frozen" if a.frozen else "dev"
    cases = load_cases(a.profile, split)
    result = score(cases, make_pipeline(a.adapter, cases))
    print(render(result, f"{a.adapter} @ {split}", show_failures=not a.frozen))
    if a.frozen:
        print("\n(frozen: per-case failures are not shown, by design)")
        append_log(log_path, now, commit, a.adapter, result)
    elif a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
