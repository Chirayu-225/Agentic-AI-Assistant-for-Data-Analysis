"""
stage0_bridge.py — the Stage 0 reader's STRUCTURE repairs, handed to the app's normal reader as repaired CSV bytes.

The experiment (eval_results/exp) found that structural file defects (junk lines above the header, footer notes, a
"Total" row, rows with an extra field) break the old reader silently, and that the Stage 0 reader (clean_read.py) closes
most of that gap. Stage 0 also cleans VALUES, and the app's own cleaner does that already; running both double-cleaned
the table. This bridge takes only the structure: it returns the file the way a careful person would fix it by hand, as
plain CSV text with every value untouched, and the normal reader and cleaner carry on from there.

Rules, all of them conservative:
  * a file with no structural defect is returned as None (the caller keeps the original bytes), so a clean file is
    read byte-for-byte as before;
  * a cheap pre-screen (one pass over the field counts, the first row and the last rows) rules out files that are
    plainly fine, so the full reader (about 1.8 s per MB) only runs for files that look suspicious;
  * the reader must be sure (status ok and no open questions), and any exception means "leave the file alone";
  * files above STAGE0_BRIDGE_MAX_BYTES (5 MB by default) are left alone: the reader needs about 28x the file size;
  * a row the reader set aside for ONE extra trailing field is kept (the extra value dropped) only if every remaining
    field has the same shape (empty, date, number or text) as its column; otherwise it stays set aside, because the
    extra comma may be inside a text value and have shifted the row;
  * the repaired text is re-parsed before it is returned; a mismatch means "leave the file alone".
Everything it did is returned as plain-English notes for the cleaning report.

Switch off with AST_STAGE0_BRIDGE=0. Used by file_reader.read_tabular_file (the app's "safe" read) only: the legacy
path, the scorer and the experiment harness do not call it.
"""
from __future__ import annotations

import collections
import csv
import io
import os

DEFAULT_MAX_BYTES = 5 * 1024 * 1024
MIN_CLASS_SHARE = 0.05      # a cell class must cover this share of a column's filled cells to count as "normal" there


def enabled() -> bool:
    return os.environ.get("AST_STAGE0_BRIDGE", "1") != "0"


def max_bytes() -> int:
    try:
        return max(int(os.environ.get("STAGE0_BRIDGE_MAX_BYTES", DEFAULT_MAX_BYTES)), 0)
    except ValueError:
        return DEFAULT_MAX_BYTES


def _plural(n: int, one: str, many: str = None) -> str:
    return f"{n:,} {one if n == 1 else (many or one + 's')}"


def _normal_classes(rows: list, width: int) -> list:
    """For each column, the set of cell classes that are normal there (empty cells are always fine)."""
    from clean_read import _cls
    out = []
    for j in range(width):
        counts = collections.Counter(c for c in (_cls(f[j]) for _, f in rows) if c)
        total = sum(counts.values())
        out.append({c for c, n in counts.items() if total and n / total >= MIN_CLASS_SHARE})
    return out


def _fits(fields: list, normal: list) -> bool:
    from clean_read import _cls
    return all((c := _cls(v)) is None or c in normal[j] for j, v in enumerate(fields))


def repair(data: bytes):
    """-> (repaired CSV bytes or None, notes [str], info {}). None means: leave the file exactly as it is."""
    info: dict = {"changed": False, "reason": ""}
    if not enabled():
        info["reason"] = "switched off"
        return None, [], info
    if len(data) > max_bytes():
        info["reason"] = "file larger than the bridge limit"
        return None, [], info
    try:
        return _repair(data, info)
    except Exception as e:  # noqa: BLE001  never let a structure repair break a file that would have read
        info["reason"] = f"reader error: {type(e).__name__}"
        return None, [], info


def _looks_clean(data: bytes) -> bool:
    """A cheap pre-screen: True only if the file is plainly a well-formed table, so the full reader can be skipped.
    Clean means: comma-separated with at least 2 columns, every row the same width, a label-like first row, no row whose
    first cell says Total/Subtotal/Sum, and a tail whose rows are mostly filled. Anything else goes to the reader."""
    from clean_read import _TOTAL, _cls
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    widths = collections.Counter()
    first = None
    tail = collections.deque(maxlen=8)
    for row in csv.reader(io.StringIO(text)):
        if not row or (len(row) == 1 and not row[0].strip()):
            continue                                         # a blank line
        if first is None:
            first = row
        widths[len(row)] += 1
        if _TOTAL.match(row[0].strip()):
            return False
        tail.append(row)
    if first is None or len(widths) != 1:
        return False
    width = next(iter(widths))
    if width < 2:
        return False                                         # a single column: probably another delimiter, let the reader look
    if any(_cls(c) in ("number", "date") for c in first) or sum(1 for c in first if c.strip()) < width / 2:
        return False                                         # the first row looks like data, or is mostly empty
    return all(sum(1 for c in r if c.strip()) >= width / 2 for r in tail)


def _repair(data: bytes, info: dict):
    from clean_read import read

    if _looks_clean(data):
        info["reason"] = "no structural defect"
        return None, [], info

    res = read(data)
    if res.get("status") != "ok" or res.get("asks"):
        info["reason"] = "the reader was not sure"
        return None, [], info
    st = res["structure"]
    header_line = st.get("header_line") or 1
    excluded = res.get("excluded", [])
    if header_line == 1 and not excluded:
        info["reason"] = "no structural defect"
        return None, [], info

    names = [c["name"] for c in res["columns"]]
    width = len(names)
    rows = list(res["rows"])                                   # [(line, fields)]
    normal = _normal_classes(rows, width)

    kept_back = collections.Counter()
    restored = 0
    for x in excluded:
        reason = x.get("reason")
        if reason == "extra_field":
            try:
                fields = next(csv.reader([x["raw"]]))
            except Exception:  # noqa: BLE001
                fields = []
            if len(fields) == width + 1 and _fits(fields[:width], normal):
                rows.append((x["line"], fields[:width]))
                restored += 1
                continue
        kept_back[reason] += 1
    rows.sort(key=lambda r: r[0])

    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(names)
    for _, fields in rows:
        w.writerow(fields)
    text = out.getvalue()

    # verify: what we are about to hand over must parse back to exactly these rows
    back = list(csv.reader(io.StringIO(text)))
    if len(back) != len(rows) + 1 or any(len(r) != width for r in back):
        info["reason"] = "the repaired text did not parse back cleanly"
        return None, [], info

    notes = []
    if header_line > 1:
        notes.append(f"Skipped {_plural(header_line - 1, 'line')} above the header: the table starts on line {header_line}.")
    set_aside = collections.Counter(x["reason"] for x in excluded if x.get("reason") != "extra_field")
    labels = {"footer_note": "footer note line", "total_row": "total row"}
    bits = [_plural(n, labels.get(r, r.replace("_", " ") + " line")) for r, n in sorted(set_aside.items())]
    if bits:
        notes.append("Set aside " + ", ".join(bits) + ": not data, so not counted.")
    if restored:
        notes.append(f"Kept {_plural(restored, 'row')} that had one extra trailing value (the extra value was dropped).")
    if kept_back.get("extra_field"):
        notes.append(f"Left out {_plural(kept_back['extra_field'], 'row')} with an extra value that may have shifted its fields.")
    info.update(changed=True, reason="repaired", header_line=header_line, set_aside=dict(set_aside),
                restored=restored, left_out=kept_back.get("extra_field", 0))
    return text.encode("utf-8"), notes, info


def bridge_bytes(data: bytes) -> bytes:
    """The repaired file, or the original bytes when there is nothing to repair (what the experiment arm scores)."""
    new, _notes, _info = repair(data)
    return new if new is not None else data
