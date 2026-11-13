"""
audit_ledger.py — the audit trail: every change the pipeline made, row by row.

HOW IT WORKS
  The pipeline (reader -> rule cleaner -> fill -> value repairs -> exact-arithmetic fill -> the agent / the user's answers)
  never has to log its own changes. After each stage the ledger compares the table before and after, cell by cell, and
  records what differs: the original row number, the column, the value before and after, and the rule that did it. So a
  rule cannot change something without it showing up here.

  Then one independent check: the ORIGINAL table against the FINAL one. Any difference that no ledger entry explains is
  reported as "unexplained" (it should always be 0).

ROW NUMBERS
  "row" is the row in the user's file as a spreadsheet shows it: the header is row 1, the first data row is row 2.

WHAT IS NOT LISTED CELL BY CELL
  A whole column read as numbers or dates (the text "$1,234.50" becoming 1234.5) is one column-level entry with examples,
  not one entry per cell. Values the conversion could not read, and every other change, are listed individually.
  At most MAX_STORED entries are kept (the counts are always exact).
"""
from __future__ import annotations

import csv
import io
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

MAX_STORED = 100_000
MAX_RECONCILE_CELLS = 600_000
SENTINEL_TEXT = {"-999", "-9999", "-99", "9999", "99999", "999999", "n/a", "na", "null", "none", "nan", "#n/a", "?", "-", "--"}


def colkey(c) -> str:
    return str(c).strip().lower().replace(" ", "_")


def _short(v, n=60) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
        return ""
    if isinstance(v, pd.Timestamp):
        return str(v.date()) if v == v.normalize() else str(v)
    if isinstance(v, float):
        return repr(float(round(v, 10)))
    t = str(v)
    return t if len(t) <= n else t[: n - 1] + "…"


def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _masks(bs: pd.Series, as_: pd.Series):
    """-> (blanked, filled, changed, parsed_count) as boolean arrays over the aligned rows."""
    if bs.dtype == as_.dtype and bs.equals(as_):                # the common case: a stage did not touch this column
        z = np.zeros(len(bs), dtype=bool)
        return z, z, z, 0
    bnull = bs.isna().to_numpy()
    anull = as_.isna().to_numpy()
    blanked = ~bnull & anull
    filled = bnull & ~anull
    both = ~bnull & ~anull
    changed = np.zeros(len(bs), dtype=bool)
    parsed = 0
    if both.any():
        b_text = not (_is_num(bs) or pd.api.types.is_datetime64_any_dtype(bs) or pd.api.types.is_bool_dtype(bs))
        if b_text and (_is_num(as_) or pd.api.types.is_datetime64_any_dtype(as_)):
            if _is_num(as_):
                # text that is already a plain number ("714.0") can be checked cell by cell; text such as "$1,234.50" is a
                # conversion this check cannot verify, so it is one column-level entry
                b = pd.to_numeric(bs, errors="coerce").to_numpy(dtype=float)
                a = pd.to_numeric(as_, errors="coerce").to_numpy(dtype=float)
                plain = both & ~np.isnan(b)
                with np.errstate(invalid="ignore"):
                    changed = plain & ~np.isclose(b, a, rtol=1e-9, atol=1e-9, equal_nan=True)
                parsed = int((both & ~plain).sum())
            else:
                parsed = int(both.sum())                   # dates read from text: a column-level conversion
        elif _is_num(bs) and _is_num(as_):
            b = pd.to_numeric(bs, errors="coerce").to_numpy(dtype=float)
            a = pd.to_numeric(as_, errors="coerce").to_numpy(dtype=float)
            with np.errstate(invalid="ignore"):
                changed = both & ~np.isclose(b, a, rtol=1e-9, atol=1e-9)
        elif pd.api.types.is_datetime64_any_dtype(bs) and pd.api.types.is_datetime64_any_dtype(as_):
            changed = both & (bs.to_numpy() != as_.to_numpy())
        elif bs.dtype == object and as_.dtype == object:
            changed = both & (bs.to_numpy() != as_.to_numpy())
        else:
            changed = both & (bs.astype(str).to_numpy() != as_.astype(str).to_numpy())
    return blanked, filled, changed, parsed


class Ledger:
    def __init__(self, raw: pd.DataFrame, source_rows=None, promoted: bool = False):
        self.entries: list = []
        self.counts: Counter = Counter()          # (stage, rule, action, column) -> n
        self.evidence: dict = {}                  # (stage, rule, column) -> text
        self.cells: set = set()                   # (orig_row, colkey) explained
        self.removed: dict = {}                   # orig_row -> reason
        self.conversions: list = []               # column-level conversions
        self.columns_dropped: list = []
        self.columns_added: list = []
        self.notes: list = []
        self.truncated = False
        self.stages: list = []
        self.rows_touched: set = set()
        self.key_col = None
        self._src = source_rows
        self.raw = self._raw_view(raw, promoted)
        self._prev = None
        self._prev_orig = None

    @classmethod
    def for_clean(cls, raw: pd.DataFrame, clean: pd.DataFrame) -> "Ledger":
        """The ledger for a raw table and the cleaner's output (which carries attrs["source_rows"])."""
        return cls(raw, clean.attrs.get("source_rows"), bool(clean.attrs.get("header_promoted")))

    @staticmethod
    def _raw_view(raw: pd.DataFrame, promoted: bool) -> pd.DataFrame:
        """The raw table as the cleaner saw it: when it promoted row 0 to the header, so do we (row labels are kept)."""
        if not promoted or len(raw) == 0:
            return raw
        view = raw.iloc[1:].copy()
        view.columns = [str(v).strip() for v in raw.iloc[0]]
        return view

    @staticmethod
    def _raw_orig(raw: pd.DataFrame) -> np.ndarray:
        idx = pd.Index(raw.index)
        return idx.to_numpy() if pd.api.types.is_integer_dtype(idx) else np.arange(len(raw))

    def _orig_of(self, frame: pd.DataFrame) -> np.ndarray:
        """Original row id of each row of a cleaner-lineage frame (its index labels are the cleaner's output positions)."""
        labels = pd.Index(frame.index)
        if self._src is None or not pd.api.types.is_integer_dtype(labels):
            return labels.to_numpy() if pd.api.types.is_integer_dtype(labels) else np.arange(len(frame))
        src = np.asarray(getattr(self._src, 'values', self._src))
        lab = labels.to_numpy()
        ok = (lab >= 0) & (lab < len(src))
        out = np.where(ok, src[np.clip(lab, 0, max(len(src) - 1, 0))], -1)
        return out

    def _pick_key(self, frame: pd.DataFrame):
        best = None
        for c in frame.columns:
            n = len(frame)
            s = frame[c]
            if n == 0 or s.isna().all():
                continue
            k = colkey(c)
            idlike = k in ("id", "row_id") or k.endswith("_id") or k.endswith("id") or "number" in k or k.endswith("_no")
            if idlike and s.nunique(dropna=True) >= 0.95 * s.notna().sum():
                best = c
                break
        self.key_col = best

    # ── recording ───────────────────────────────────────────────────────────────────────────────────────────────
    def _add(self, stage, rule, action, row, column, before, after, key=""):
        self.counts[(stage, rule, action, column)] += 1
        if column is not None:
            self.cells.add((int(row), colkey(column)))
        self.rows_touched.add(int(row))
        if len(self.entries) < MAX_STORED:
            self.entries.append({"stage": stage, "rule": rule, "action": action, "row": int(row) + 2, "key": key,
                                 "column": column, "before": _short(before), "after": _short(after)})
        else:
            self.truncated = True

    def _key_for(self, frame: pd.DataFrame, positions: np.ndarray) -> list:
        if self.key_col is None or self.key_col not in frame.columns:
            return [""] * len(positions)
        col = frame[self.key_col].iloc[positions]
        return [_short(v, 40) for v in col.to_numpy()]

    # ── a stage ─────────────────────────────────────────────────────────────────────────────────────────────────
    def start(self, clean: pd.DataFrame):
        """The first stage: raw table -> rule-cleaned table."""
        raw = self.raw
        self._pick_key(clean)
        self._diff("Rule cleaning", raw, clean, self._raw_orig(raw), self._orig_of(clean), rule_fn=self._clean_rule,
                   normalise_names=True, first=True)
        self._prev, self._prev_orig = clean, self._orig_of(clean)

    def advance(self, stage: str, after: pd.DataFrame, rules: dict | None = None, default_rule: str = "",
                removed_reason: str = "", evidence: dict | None = None):
        """A later stage: the previous table -> `after`. rules maps column -> rule name (from that stage's own report)."""
        a_orig = self._orig_of(after)
        rules = rules or {}

        def rule_fn(col, action, b, a):
            named = rules.get((col, action)) or rules.get(col) or rules.get(colkey(col))
            if named:
                return named
            if action == "changed":  # the user's "divide by 1000" answer (scale_review.py)
                try:
                    if float(b) >= 1000 and float(b) % 1000 == 0 and float(a) == float(b) / 1000:
                        return "Divided by 1000 (your choice)"
                except (TypeError, ValueError):
                    pass
            return default_rule or stage

        self._diff(stage, self._prev, after, self._prev_orig, a_orig, rule_fn=rule_fn, removed_reason=removed_reason,
                   evidence=evidence or {})
        self._prev, self._prev_orig = after, a_orig

    def _clean_rule(self, col, action, b, a):
        if action == "blanked":
            t = str(b).strip().lower()
            return "Placeholder value (sentinel) replaced by blank" if t in SENTINEL_TEXT else "Value could not be read, left blank"
        if action == "filled":
            return "Blank filled"
        sb, sa = str(b), str(a)
        if sb.strip().casefold() == sa.strip().casefold():
            return "Spacing or case standardised"
        return "Spelling variant merged"

    def _diff(self, stage, before, after, b_orig, a_orig, rule_fn, normalise_names=False, first=False,
              removed_reason="", evidence=None):
        if stage not in self.stages:
            self.stages.append(stage)
        ev = evidence or {}
        bk = {colkey(c): c for c in before.columns} if normalise_names else {str(c): c for c in before.columns}
        ak = {colkey(c): c for c in after.columns} if normalise_names else {str(c): c for c in after.columns}
        if len(bk) != len(before.columns) or len(ak) != len(after.columns):
            self.notes.append(f"{stage}: duplicate column names, so this stage's cell changes are not listed.")
            return
        for k, c in bk.items():
            if k not in ak:
                self.columns_dropped.append({"stage": stage, "column": str(c), "reason": "removed"})
        for k, c in ak.items():
            if k not in bk:
                self.columns_added.append({"stage": stage, "column": str(c)})
        bpos = pd.Series(np.arange(len(before)), index=b_orig)
        apos = pd.Series(np.arange(len(after)), index=a_orig)
        if bpos.index.has_duplicates or apos.index.has_duplicates:
            self.notes.append(f"{stage}: row numbers are not unique, so this stage's changes are not listed.")
            return
        common = bpos.index.intersection(apos.index)
        removed_ids = bpos.index.difference(apos.index)
        bi = bpos.loc[common].to_numpy()
        ai = apos.loc[common].to_numpy()
        order = np.argsort(common.to_numpy(), kind="stable")
        bi, ai, rows = bi[order], ai[order], common.to_numpy()[order]
        if len(removed_ids):
            self._record_removed(stage, before, bpos, removed_ids, apos, after, removed_reason)
        keys = self._key_for(after, ai)
        for k, bc in bk.items():
            if k not in ak:
                continue
            ac = ak[k]
            try:
                bs = before[bc].iloc[bi]
                as_ = after[ac].iloc[ai]
                blanked, filled, changed, parsed = _masks(bs, as_)
            except Exception:  # noqa: BLE001  one odd column must not break the audit
                self.notes.append(f"{stage}: column '{ac}' could not be compared.")
                continue
            if parsed:
                kind = "dates" if pd.api.types.is_datetime64_any_dtype(as_) else "numbers"
                self.conversions.append({"stage": stage, "column": str(ac), "count": parsed, "to": kind})
            col_name = str(ac)
            bvals, avals = bs.to_numpy(dtype=object), as_.to_numpy(dtype=object)
            for action, mask in (("blanked", blanked), ("filled", filled), ("changed", changed)):
                for j in np.nonzero(mask)[0]:
                    rule = rule_fn(col_name, action, bvals[j], avals[j])
                    self._add(stage, rule, action, rows[j], col_name, bvals[j], avals[j], keys[j])
                    if (stage, rule, col_name) not in self.evidence and (stage, rule, col_name) in ev:
                        self.evidence[(stage, rule, col_name)] = ev[(stage, rule, col_name)]

    def _record_removed(self, stage, before, bpos, removed_ids, apos, after, reason):
        key_b = None
        if self.key_col is not None:
            for c in before.columns:
                if colkey(c) == colkey(self.key_col):
                    key_b = c
                    break
        kept_ids = apos.index
        full_b = None
        first_stage = stage == "Rule cleaning"
        try:
            full_b = pd.util.hash_pandas_object(before.astype(str), index=False).to_numpy()
        except Exception:  # noqa: BLE001
            full_b = None
        hash_to_kept = {}
        key_to_kept = {}
        kp = bpos.loc[kept_ids].to_numpy() if len(kept_ids) else np.array([], dtype=int)
        for pos, oid in zip(kp, kept_ids):
            if full_b is not None:
                hash_to_kept.setdefault(int(full_b[pos]), int(oid))
            if key_b is not None and first_stage:
                key_to_kept.setdefault(str(before[key_b].iloc[pos]), int(oid))
        for oid in removed_ids:
            pos = int(bpos.loc[oid])
            rule, detail = reason or "Row removed", ""
            if full_b is not None and int(full_b[pos]) in hash_to_kept:
                rule = "Exact duplicate removed" if first_stage else "Exact duplicate removed (your choice)"
                detail = f"copy of row {hash_to_kept[int(full_b[pos])] + 2}"
            elif first_stage:
                if key_b is not None and str(before[key_b].iloc[pos]) in key_to_kept:
                    rule = "Repeated record (same ID) removed"
                    detail = f"same {self.key_col} as row {key_to_kept[str(before[key_b].iloc[pos])] + 2}; the more complete copy was kept"
                else:
                    rule = "Row removed by the cleaner"
            if key_b is not None:
                key = _short(before[key_b].iloc[pos], 40)
            else:  # no ID column: show the start of the row so it can be recognised
                key = " | ".join(t for t in (_short(v, 16) for v in before.iloc[pos].tolist()[:3]) if t)
            self.removed[int(oid)] = rule
            self.rows_touched.add(int(oid))
            self.counts[(stage, rule, "removed", None)] += 1
            if len(self.entries) < MAX_STORED:
                self.entries.append({"stage": stage, "rule": rule, "action": "removed", "row": int(oid) + 2, "key": key,
                                     "column": None, "before": "(whole row)", "after": detail})
            else:
                self.truncated = True

    # ── the independent check ───────────────────────────────────────────────────────────────────────────────────
    def reconcile(self, final: pd.DataFrame) -> dict:
        """Original table vs FINAL table. Anything that differs and no ledger entry explains is 'unexplained'."""
        raw = self.raw
        f_orig = self._orig_of(final)
        r_orig = self._raw_orig(raw)
        rk = {colkey(c): c for c in raw.columns}
        fk = {colkey(c): c for c in final.columns}
        if len(rk) != len(raw.columns) or len(fk) != len(final.columns):
            return {"count": 0, "examples": [], "checked": False, "reason": "duplicate column names"}
        rpos = pd.Series(np.arange(len(raw)), index=r_orig)
        fpos = pd.Series(np.arange(len(final)), index=f_orig)
        if rpos.index.has_duplicates or fpos.index.has_duplicates:
            return {"count": 0, "examples": [], "checked": False, "reason": "row numbers are not unique"}
        removed = rpos.index.difference(fpos.index)
        unexplained_rows = [int(r) for r in removed if int(r) not in self.removed]
        common = rpos.index.intersection(fpos.index).to_numpy()
        order = np.argsort(common, kind="stable")
        common = common[order]
        bi = rpos.loc[common].to_numpy()
        ai = fpos.loc[common].to_numpy()
        bad: list = []
        total_cells = 0
        for k, rc in rk.items():
            if k not in fk:
                continue
            try:
                bs = raw[rc].iloc[bi]
                as_ = final[fk[k]].iloc[ai]
                blanked, filled, changed, _ = _masks(bs, as_)
            except Exception:  # noqa: BLE001
                continue
            m = blanked | filled | changed
            total_cells += int(m.sum())
            if total_cells > MAX_RECONCILE_CELLS:
                return {"count": 0, "examples": [], "checked": False, "reason": "too many changes to cross-check cell by cell"}
            for j in np.nonzero(m)[0]:
                if (int(common[j]), k) not in self.cells:
                    bad.append({"row": int(common[j]) + 2, "column": str(fk[k]), "before": _short(bs.iloc[j]), "after": _short(as_.iloc[j])})
        for r in unexplained_rows[:20]:
            bad.append({"row": r + 2, "column": None, "before": "(whole row)", "after": "missing from the final table"})
        return {"count": len(bad) + max(len(unexplained_rows) - 20, 0), "examples": bad[:10], "checked": True, "reason": ""}

    # ── duplicates still in the table ───────────────────────────────────────────────────────────────────────────
    def duplicate_groups(self, final: pd.DataFrame, columns: list | None = None, max_groups: int = 25) -> dict:
        cols = [c for c in (columns or list(final.columns)) if c in final.columns]
        if not cols or len(final) == 0:
            return {"groups": 0, "rows": 0, "extra": 0, "sample": []}
        h = pd.util.hash_pandas_object(final[cols], index=False)
        dup = h.duplicated(keep=False)
        if not dup.any():
            return {"groups": 0, "rows": 0, "extra": 0, "sample": []}
        orig = self._orig_of(final)
        sub = pd.DataFrame({"h": h[dup].to_numpy(), "orig": orig[dup.to_numpy()], "pos": np.nonzero(dup.to_numpy())[0]})
        groups = [g for _, g in sub.groupby("h", sort=False)]
        groups.sort(key=lambda g: (-len(g), int(g["orig"].min())))
        sample = []
        for g in groups[:max_groups]:
            first_pos = int(g["pos"].iloc[0])
            vals = {str(c): _short(final[c].iloc[first_pos], 30) for c in cols[:8]}
            sample.append({"rows": [int(o) + 2 for o in sorted(g["orig"])][:30], "count": int(len(g)), "values": vals})
        return {"groups": len(groups), "rows": int(dup.sum()), "extra": int(dup.sum()) - len(groups), "sample": sample}

    def release(self):
        """Drop the big frames once the summary is built (the entries are all the dashboard and the CSV need)."""
        self.raw = None
        self._prev = None
        self._prev_orig = None

    # ── output ──────────────────────────────────────────────────────────────────────────────────────────────────
    def summary(self, final: pd.DataFrame, dup_columns: list | None = None) -> dict:
        by_action = Counter()
        for (stage, rule, action, col), n in self.counts.items():
            by_action[action] += n
        by_rule = []
        for (stage, rule, action, col), n in sorted(self.counts.items(), key=lambda kv: -kv[1]):
            by_rule.append({"stage": stage, "rule": rule, "action": action, "column": col, "count": int(n),
                            "evidence": self.evidence.get((stage, rule, col), "")})
        rec = self.reconcile(final)
        return {
            "totals": {"cellsChanged": int(by_action["changed"]), "cellsBlanked": int(by_action["blanked"]),
                       "cellsFilled": int(by_action["filled"]), "rowsRemoved": int(by_action["removed"]),
                       "rowsTouched": len(self.rows_touched), "rowsRaw": int(len(self.raw)), "rowsFinal": int(len(final)),
                       "columnsDropped": [d["column"] for d in self.columns_dropped],
                       "columnsAdded": [d["column"] for d in self.columns_added]},
            "byRule": by_rule[:300],
            "conversions": self.conversions,
            "stages": self.stages,
            "unexplained": rec,
            "duplicates": self.duplicate_groups(final, dup_columns),
            "entryCount": len(self.entries),
            "truncated": self.truncated,
            "notes": self.notes,
        }

    def page(self, offset=0, limit=100, rule=None, column=None, action=None, q=None) -> dict:
        rows = self.entries
        if rule:
            rows = [e for e in rows if e["rule"] == rule]
        if column:
            rows = [e for e in rows if e["column"] == column]
        if action:
            rows = [e for e in rows if e["action"] == action]
        if q:
            t = str(q).strip().lower()
            rows = [e for e in rows if t in str(e["row"]) or t in e["key"].lower() or t in e["before"].lower() or t in e["after"].lower()]
        offset = max(int(offset), 0)
        limit = min(max(int(limit), 1), 500)
        return {"total": len(rows), "offset": offset, "entries": rows[offset: offset + limit]}

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["stage", "rule", "action", "row", "key", "column", "before", "after"])
        for e in self.entries:
            w.writerow([e["stage"], e["rule"], e["action"], e["row"], e["key"], e["column"] or "", e["before"], e["after"]])
        return buf.getvalue()


# ── a small in-memory store so paging and the CSV download do not rebuild the ledger ─────────────────────────────────
_TTL = 30 * 60
_MAX = 4
_store: "OrderedDict" = OrderedDict()


def store_put(key, ledger: Ledger, summary: dict):
    import time
    _store[key] = (time.time(), ledger, summary)
    _store.move_to_end(key)
    while len(_store) > _MAX:
        _store.popitem(last=False)


def store_get(key):
    import time
    hit = _store.get(key)
    if hit is None:
        return None
    if time.time() - hit[0] > _TTL:
        _store.pop(key, None)
        return None
    return hit[1], hit[2]


def store_drop(prefix):
    for k in [k for k in _store if k[: len(prefix)] == prefix]:
        _store.pop(k, None)
