"""
test_clean_bench.py — is the cleaning benchmark's ground truth itself correct?
A benchmark with wrong labels is worse than none, so these checks verify the
generator against the files it writes: every label's (line, column) really
holds the value the label says; each row carries at most one injected defect;
decoys never overlap defects; dev and frozen vocabularies are disjoint; the
generator is deterministic and shares no code with the pipeline; the manifest
tripwire fires on change. Stdlib only.   Run: python test_clean_bench.py
"""
import ast
import csv
import datetime as dt
import io
import json
import os
import re
import sys
import tempfile
from decimal import Decimal

import clean_bench as cb

FAIL = []
def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {str(detail)[:300]}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)

HERE = os.path.dirname(os.path.abspath(__file__))
BUILT = {}
def built(profile, split):
    key = (profile, split)
    if key not in BUILT:
        BUILT[key] = [(s, *cb.build_case(s)) for s in cb.specs_for(profile, split)]
    return BUILT[key]

def read_lines(files):
    out, rd = {}, csv.reader(io.StringIO(files["data.csv"].decode("utf-8-sig")))
    for row in rd:
        out[rd.line_num] = row
    return out

# ── independence from the pipeline ───────────────────────────────────────────
src = open(os.path.join(HERE, "clean_bench.py"), encoding="utf-8").read()
mods = set()
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Import):
        mods |= {a.name.split(".")[0] for a in node.names}
    elif isinstance(node, ast.ImportFrom):
        mods.add((node.module or "").split(".")[0])
STDLIB = {"__future__", "argparse", "csv", "datetime", "hashlib", "io", "json", "os", "random", "sys", "dataclasses", "decimal", "typing"}
check("the generator imports only the standard library (no project code)", mods <= STDLIB, sorted(mods - STDLIB))

# frozen-only vocabulary must not already live in the detectors' source
frozen_tokens = set()
fv = cb.VOCAB["frozen"]
for slot in ("sentinels", "numeric_sentinels", "date_formats"):
    frozen_tokens |= set(fv[slot])
frozen_tokens |= {t for line in fv["title_lines"] for t in line}
for pair in fv["bool_encodings"]:
    frozen_tokens |= set(pair)
leaks = {}
for fname in ("cleaning.py", "cleaning_agent.py", "profiler.py", "pii_redaction.py", "file_reader.py", "table_extraction.py", "clean_schema.py",
              "clean_read.py", "clean_adapter_stage0.py"):
    p = os.path.join(HERE, fname)
    if os.path.exists(p):
        text = open(p, encoding="utf-8").read()
        for tok in frozen_tokens:
            if len(tok) >= 3 and re.search(r"[\"']" + re.escape(tok) + r"[\"']", text):
                leaks.setdefault(fname, []).append(tok)
check("no frozen-only vocabulary appears as a literal in detector sources", not leaks, leaks)

# ── vocabularies and shapes ──────────────────────────────────────────────────
for slot in cb.VOCAB["dev"]:
    a = {json.dumps(x) for x in cb.VOCAB["dev"][slot]}
    b = {json.dumps(x) for x in cb.VOCAB["frozen"][slot]}
    check(f"dev and frozen share no '{slot}'", not (a & b), a & b)
check("every locale named in a split exists", all(l in cb.LOCALES for s in cb.VOCAB.values() for l in s["locales"]))
dev_locs = {t[2] for t in cb.PROFILES["full"]["dev"]} - {"plain"}
fro_locs = {t[2] for t in cb.PROFILES["full"]["frozen"]} - {"plain"}
check("frozen uses locales dev never uses", not (dev_locs & fro_locs) and fro_locs, (dev_locs, fro_locs))
dev_sizes = {t[1] for t in cb.PROFILES["full"]["dev"]}
fro_sizes = {t[1] for t in cb.PROFILES["full"]["frozen"]}
check("frozen uses table sizes dev never uses", not (dev_sizes & fro_sizes), dev_sizes & fro_sizes)
check("sizes span about 50 rows to tens of thousands", min(dev_sizes) <= 60 and max(fro_sizes) >= 30000)
for split in ("dev", "frozen"):
    ctrl = [t for t in cb.PROFILES["full"][split] if t[3] == 0]
    check(f"{split}: has a 0%-defect control for every family", {t[0] for t in ctrl} == {"orders", "staff", "inventory"})

# ── determinism ──────────────────────────────────────────────────────────────
m1, m2 = cb.build("mini"), cb.build("mini")
check("building twice gives byte-identical output (hash equal)", m1 == m2)
check("dev and frozen are different data", m1["splits"]["dev"]["sha256"] != m1["splits"]["frozen"]["sha256"])

# ── ground truth vs the rendered file ────────────────────────────────────────
EXPECTED = {"fix", "ask", "quarantine", "flag", "none"}
LABEL_KEYS = ["id", "kind", "defect", "scope", "line", "cells", "expected_action", "accepted_actions",
              "accepted_fixes", "composition", "related_line", "extra", "note"]
bad_coord, bad_schema, bad_double, bad_decoy, bad_expected = [], [], [], [], []
for profile in ("mini", "coverage"):
    for split in ("dev", "frozen"):
        for spec, files, doc in built(profile, split):
            lines, labels = read_lines(files), doc["labels"]
            seen_lines = {}
            for L in labels:
                if list(L) != LABEL_KEYS or L["expected_action"] not in EXPECTED or L["kind"] not in ("defect", "decoy"):
                    bad_schema.append((spec.case_id, L["id"]))
                if L["kind"] == "decoy" and L["expected_action"] != "none":
                    bad_decoy.append((spec.case_id, L["id"]))
                if L["kind"] == "defect" and L["expected_action"] not in L["accepted_actions"]:
                    bad_expected.append((spec.case_id, L["id"]))
                if L["scope"] in ("cell", "row"):
                    if L["line"] is None or L["line"] not in lines:
                        bad_coord.append((spec.case_id, L["id"], "no such line"))
                        continue
                    seen_lines.setdefault(L["line"], []).append(L)
                    for c in L["cells"]:
                        row = lines[L["line"]]
                        got = row[c["col_index"] - 1] if c["col_index"] - 1 < len(row) else None
                        if got != c["shown"]:
                            bad_coord.append((spec.case_id, L["id"], L["defect"], got, c["shown"]))
            for ln, ls in seen_lines.items():
                if len(ls) > 1:
                    bad_double.append((spec.case_id, ln, [x["defect"] for x in ls]))
check("every cell/row label's (line, column) holds exactly the value it records", not bad_coord, bad_coord[:3])
check("every label has the full schema and a valid action vocabulary", not bad_schema, bad_schema[:3])
check("decoys are always expected_action 'none'", not bad_decoy)
check("a defect's expected action is among its accepted actions", not bad_expected, bad_expected[:3])
check("no physical line carries two injected labels (one defect per row)", not bad_double, bad_double[:3])

# truth.csv alignment and clean values
bad_truth, bad_clean = [], []
for spec, files, doc in built("mini", "dev") + built("mini", "frozen") + built("coverage", "dev"):
    lines = read_lines(files)
    truth = list(csv.DictReader(io.StringIO(files["truth.csv"].decode("utf-8"))))
    tl = {int(t["__line"]): t for t in truth}
    if len(tl) != doc["meta"]["n_data_rows"]:
        bad_truth.append((spec.case_id, "row count"))
    for L in doc["labels"]:
        if L["scope"] == "cell" and L["kind"] == "defect":
            for c in L["cells"]:
                t = tl.get(L["line"])
                if t is None:
                    bad_clean.append((spec.case_id, L["id"], "defect on a row absent from truth"))
                elif (t[c["column"]] or None) != c["clean"]:
                    bad_clean.append((spec.case_id, L["id"], L["defect"], t[c["column"]], c["clean"]))
        if L["scope"] == "row" and L["defect"] in ("dup_exact", "dup_near", "total_row", "ragged_row", "blank_row", "footer_note"):
            if L["defect"] in ("dup_exact", "dup_near", "total_row", "blank_row", "footer_note") and L["line"] in tl:
                bad_truth.append((spec.case_id, L["defect"], "extra/structural row leaked into truth"))
check("truth.csv has one row per original data row, keyed by its line", not bad_truth, bad_truth[:3])
check("every defect's recorded clean value equals truth.csv", not bad_clean, bad_clean[:3])

# decoys and controls
for split in ("dev", "frozen"):
    allc = built("mini", split) + built("coverage", split)
    check(f"{split}: every case carries decoys", all(any(L["kind"] == "decoy" for L in d["labels"]) for _, _, d in allc))
ctrl = [x for x in built("full" if False else "mini", "dev") if x[0].rate == 0]
check("0%-defect controls contain no defect labels", all(L["kind"] == "decoy" for _, _, d in ctrl for L in d["labels"]), [d["meta"]["case_id"] for _, _, d in ctrl])

# decoys really are legitimate
dec_ok = True
for spec, files, doc in built("coverage", "dev"):
    tl = {int(t["__line"]): t for t in csv.DictReader(io.StringIO(files["truth.csv"].decode("utf-8")))}
    for L in doc["labels"]:
        if L["defect"] == "same_attributes_distinct_key":
            a, b = tl[L["line"]], tl[L["related_line"]]
            key = L["cells"][0]["column"]
            same = all(a[k] == b[k] for k in a if k not in (key, "__line"))
            dec_ok &= same and a[key] != b[key]
        if L["defect"] == "refund_negative_amount":
            dec_ok &= tl[L["line"]]["status"] == "Refunded" and Decimal(tl[L["line"]]["amount"]) < 0
        if L["defect"] == "zero_stock_valid":
            dec_ok &= tl[L["line"]]["on_hand"] == "0" and tl[L["line"]]["in_stock"] == "false"
check("decoys are valid in truth: same-but-key records, refunds, zero stock", dec_ok)

# structural defects
for split in ("dev", "frozen"):
    for spec, files, doc in built("mini", split):
        b, lines, meta = spec.bundle, read_lines(files), doc["meta"]
        names = [c["name"] for c in meta["columns"]]
        check(f"{spec.case_id}: BOM present iff declared", files["data.csv"].startswith(b"\xef\xbb\xbf") == bool(b.get("bom")))
        hl = b.get("header_line", 1)
        want = names + ([""] if b.get("trailing_delim") else [])
        check(f"{spec.case_id}: header sits on line {hl}", lines[hl] == want, lines.get(hl))
        if hl > 1:
            above = [lines[i] for i in range(1, hl)]
            padded = cb.VOCAB[split]["title_pad"][0]
            check(f"{spec.case_id}: the {hl - 1} line(s) above the header are title lines" + (" padded to the table width" if padded else ""),
                  all(len(r) == len(names) for r in above if r or padded) if padded else all(len(r) <= 2 for r in above), above)
        for L in doc["labels"]:
            if L["defect"] == "ragged_row":
                actual = len(lines[L["line"]]) - (1 if b.get("trailing_delim") else 0)
                check(f"{spec.case_id}: ragged row at line {L['line']} really has {L['extra']['fields_found']} fields",
                      actual == L["extra"]["fields_found"] and actual != L["extra"]["fields_expected"], actual)
            if L["defect"] == "dup_exact":
                check(f"{spec.case_id}: exact duplicate at line {L['line']} equals line {L['related_line']}", lines[L["line"]] == lines[L["related_line"]])
            if L["defect"] == "dup_near":
                a, o = lines[L["line"]], lines[L["related_line"]]
                diff = [i for i in range(len(a)) if a[i] != o[i]]
                check(f"{spec.case_id}: near duplicate differs only in '{L['extra']['changed_column']}'",
                      len(diff) == 1 and names[diff[0]] == L["extra"]["changed_column"] and a[diff[0]].strip().lower() == o[diff[0]].strip().lower())
            if L["defect"] == "blank_row":
                check(f"{spec.case_id}: blank row at line {L['line']} is blank", lines[L["line"]] == [])
            if L["defect"] == "total_row":
                tl = csv.DictReader(io.StringIO(files["truth.csv"].decode("utf-8")))
                col = L["extra"]["sums"]
                s = sum((Decimal(t[col]) for t in tl), Decimal("0"))
                check(f"{spec.case_id}: TOTAL equals the sum of the original rows", Decimal(L["cells"][0]["clean"]) == s and lines[L["line"]][0] == "TOTAL")
            if L["defect"] == "footer_note":
                check(f"{spec.case_id}: footer line comes after all data", L["line"] > max(l for l, r in lines.items() if r and r[0].startswith(("ORD-", "E0", "SKU-"))))

# orphan keys are really orphans against the companion table
for split in ("dev", "frozen"):
    for spec, files, doc in built("coverage", split):
        if spec.family != "orders":
            continue
        ids = {r["customer_id"] for r in csv.DictReader(io.StringIO(files["customers.csv"].decode("utf-8")))}
        orphans = [L for L in doc["labels"] if L["defect"] == "orphan_fk"]
        check(f"{spec.case_id}: orphan keys are absent from customers.csv and their clean ids are present",
              bool(orphans) and all(L["cells"][0]["shown"] not in ids and L["cells"][0]["clean"] in ids for L in orphans))
        valid = {r["customer_id"] for r in csv.DictReader(io.StringIO(files["truth.csv"].decode("utf-8")))}
        check(f"{spec.case_id}: every non-orphan customer id in truth exists", valid <= ids)

# dates: ambiguity is real where claimed and absent where not
for split in ("dev", "frozen"):
    for spec, files, doc in built("mini", split):
        tl = list(csv.DictReader(io.StringIO(files["truth.csv"].decode("utf-8"))))
        dcols = [c["name"] for c in doc["meta"]["columns"] if c["kind"] == "date"]
        days = [int(r[c][8:10]) for r in tl for c in dcols if r[c]]
        amb = [L for L in doc["labels"] if L["defect"] == "date_ambiguous"]
        if spec.flags.get("ambig"):
            check(f"{spec.case_id}: ambiguous file has day<=12 in every date and a 'date_ambiguous' label per date column",
                  max(days) <= 12 and len(amb) == len(dcols) and all(L["expected_action"] == "ask" for L in amb))
        else:
            check(f"{spec.case_id}: unambiguous file has no ambiguity labels and some day>12", not amb and max(days) > 12)

# vocabulary actually used by each split stays inside that split's vocabulary
for split, other in (("dev", "frozen"), ("frozen", "dev")):
    v, ov = cb.VOCAB[split], cb.VOCAB[other]
    used_sent, used_fmt, used_ws, used_enc, used_units = set(), set(), set(), [], set()
    for spec, files, doc in built("coverage", split):
        for L in doc["labels"]:
            c = L["cells"][0] if L["cells"] else None
            if L["defect"] == "sentinel_null":
                used_sent.add(c["shown"])
            if L["defect"] == "date_mixed_format":
                used_fmt.add(L["extra"]["format"])
            if L["defect"] == "whitespace":
                used_ws |= set(ch for ch in c["shown"] if ch.isspace())
            if L["defect"] == "unit_mismatch":
                used_units.add(L["extra"]["unit_shown"])
    check(f"{split}: sentinels used come only from the {split} vocabulary", used_sent and used_sent <= set(v["sentinels"] + v["numeric_sentinels"]), used_sent)
    check(f"{split}: mixed date formats come only from the {split} vocabulary", used_fmt and used_fmt <= set(v["date_formats"]), used_fmt)
    check(f"{split}: whitespace characters come only from the {split} vocabulary", used_ws and used_ws <= set("".join(v["whitespace"])), used_ws)
    check(f"{split}: alternative units come only from the {split} unit pairs", used_units and used_units <= {p[1] for p in v["unit_pairs"]}, used_units)
    check(f"{split}: nothing from the {other} vocabulary leaked in", not (used_sent & set(ov["sentinels"] + ov["numeric_sentinels"])) and not (used_fmt & set(ov["date_formats"])))

# coverage: every defect type appears often enough to score, in both splits
WANT = {d.name for d in cb.DEFECTS} - {"compound_money", "compound_category"} | {"compound"}
for split in ("dev", "frozen"):
    counts = {}
    for _, _, doc in built("coverage", split):
        for k, n in doc["meta"]["label_counts"].items():
            counts[k] = counts.get(k, 0) + n
    thin = {d: counts.get(f"defect:{d}", 0) for d in WANT if counts.get(f"defect:{d}", 0) < 5}
    check(f"{split}: every cell/row defect type has >= 5 samples", not thin, thin)
    decoys = {k for k in counts if k.startswith("decoy:")}
    check(f"{split}: all decoy types are present", {"decoy:heavy_tail_valid", "decoy:rare_valid_category", "decoy:refund_negative_amount",
          "decoy:zero_stock_valid", "decoy:same_attributes_distinct_key", "decoy:optional_null"} <= decoys, decoys)
    comp = {tuple(L["composition"]) for _, _, doc in built("coverage", split) for L in doc["labels"] if L["defect"] == "compound"}
    check(f"{split}: compound defects record their composition order", comp == {("number_as_text", "whitespace"), ("category_variant", "whitespace")}, comp)
    check(f"{split}: column-level conditions appear", all(counts.get(f"defect:{d}", 0) >= 1 for d in ("number_formatted_text", "date_locale_format")) or split == "dev", counts)
    struct = {}
    for _, _, doc in built("mini", split):
        for k, n in doc["meta"]["label_counts"].items():
            struct[k] = struct.get(k, 0) + n
    need = {"bom", "header_offset", "footer_note", "blank_row", "total_row", "ragged_row", "boolean_mixed_encoding"} if split == "dev" else \
           {"bom", "header_offset", "footer_note", "blank_row", "total_row", "ragged_row", "trailing_delimiter", "boolean_mixed_encoding"}
    check(f"{split}: structural defects appear in the mini profile", all(struct.get(f"defect:{k}", 0) >= 1 for k in need), {k: struct.get(f"defect:{k}", 0) for k in need})

# ── round trip: the recorded clean value must be recoverable from what is shown ──
# Labels and truth.csv both derive from the generator's clean values, so they can agree and still
# be wrong together. These checks use parsers written HERE, independent of the generator.
def parse_date_by(fmt, text):
    names = {m: i + 1 for i, m in enumerate(cb.MONTHS)}
    rx = re.escape(fmt)
    for tok, pat in (("%Y", r"(?P<Y>\d{4})"), ("%m", r"(?P<m>\d{2})"), ("%d", r"(?P<d>\d{2})"),
                     ("%B", r"(?P<B>[A-Za-z]+)"), ("%b", r"(?P<b>[A-Za-z]{3})")):
        rx = rx.replace(tok, pat)
    m = re.fullmatch(rx, text)
    if not m:
        return None
    g = m.groupdict()
    month = int(g["m"]) if "m" in g else names[g["B"]] if "B" in g else {k[:3]: v for k, v in names.items()}[g["b"]]
    return dt.date(int(g["Y"]), month, int(g["d"])).isoformat()

def digits_only(text):
    return Decimal(re.sub(r"[^0-9.\-]", "", text))

rt_bad, rt_seen = [], {}
for profile in ("mini", "coverage"):
    for split in ("dev", "frozen"):
        for spec, files, doc in built(profile, split):
            for L in doc["labels"]:
                if L["kind"] != "defect" or not L["cells"] or not (L["scope"] == "cell" or L["defect"] == "swapped_values"):
                    continue
                d, cs_ = L["defect"], L["cells"]
                c = cs_[0]
                sh, cl = c["shown"], c["clean"]
                rt_seen[d] = rt_seen.get(d, 0) + 1
                if d != "swapped_values" and (cl is None or cl == sh):
                    rt_bad.append((spec.case_id, L["id"], d, "clean missing or equal to shown"))
                    continue
                ok = True
                if d == "whitespace":
                    ok = sh.strip() == cl
                elif d in ("number_as_text", "compound") and "number_as_text" in (L["composition"] or ["number_as_text"]):
                    ok = digits_only(sh) == Decimal(cl)
                elif d == "percent_scale":
                    ok = Decimal(sh) / 100 == Decimal(cl)
                elif d == "lost_leading_zero":
                    ok = sh == cl.lstrip("0") and cl.startswith("0")
                elif d == "date_mixed_format":
                    ok = parse_date_by(L["extra"]["format"], sh) == cl
                elif d == "unit_mismatch":
                    factor = next(p[2] for p in cb.VOCAB[split]["unit_pairs"] if p[1] == L["extra"]["unit_shown"])
                    ok = Decimal(sh.split()[0]) / factor == Decimal(cl)
                elif d == "swapped_values":
                    ok = cs_[0]["shown"] == cs_[1]["clean"] and cs_[1]["shown"] == cs_[0]["clean"]
                elif d == "future_date":
                    ok = L["extra"]["year"] >= 2031 and str(L["extra"]["year"]) in sh
                elif d == "impossible_value":
                    ok = any(f == {c["column"]: None} for f in L["accepted_fixes"]) and (sh.startswith("-") or sh in ("0", "7", "9", "10", "99"))
                elif d == "orphan_fk":
                    ok = any(f == {c["column"]: cl} for f in L["accepted_fixes"])
                if not ok:
                    rt_bad.append((spec.case_id, L["id"], d, sh, cl))
                if any("unfixed" not in x for x in cs_):
                    rt_bad.append((spec.case_id, L["id"], d, "cell lacks 'unfixed'"))
                u = c["unfixed"]
                if d == "outlier_magnitude" and Decimal(u) != Decimal(cl) * 100:
                    rt_bad.append((spec.case_id, L["id"], d, "unfixed is not clean x 100", u, cl))
                if d == "future_date" and not u.startswith(str(L["extra"]["year"])):
                    rt_bad.append((spec.case_id, L["id"], d, "unfixed is not the future date", u))
                if d == "end_before_start" and not (u < cl):
                    rt_bad.append((spec.case_id, L["id"], d, "unfixed date is not before the clean date", u, cl))
                if d == "impossible_value" and u is None:
                    rt_bad.append((spec.case_id, L["id"], d, "missing unfixed"))
                if d in ("whitespace", "number_as_text", "date_mixed_format", "category_variant", "sentinel_null") and u is not None:
                    rt_bad.append((spec.case_id, L["id"], d, "reformatting IS the fix here, so unfixed must be null"))
                # every accepted fix must name a column this label covers
                cols = {x["column"] for x in cs_}
                if any(set(f) - cols for f in L["accepted_fixes"]):
                    rt_bad.append((spec.case_id, L["id"], d, "fix names a foreign column"))
check("every defect's clean value is recoverable from its shown value (independent parsers)", not rt_bad, rt_bad[:3])
check("round-trip covers every parseable defect type", {"whitespace", "number_as_text", "percent_scale", "lost_leading_zero",
      "date_mixed_format", "unit_mismatch", "swapped_values", "future_date", "impossible_value", "orphan_fk", "compound"} <= set(rt_seen), sorted(rt_seen))

# ── the FULL profile must give every defect type enough samples to score ─────
full = cb.build("full")
FILE_LEVEL = {"bom", "header_offset", "total_row", "trailing_delimiter", "footer_note", "boolean_mixed_encoding",
              "date_ambiguous", "date_locale_format", "number_formatted_text"}  # one per case or column: counts are small by nature
for split in ("dev", "frozen"):
    lc = full["splits"][split]["label_counts"]
    cell_row = {d: lc.get(f"defect:{d}", 0) for d in WANT if d not in FILE_LEVEL | {"blank_row", "ragged_row"}}
    thin = {d: n for d, n in cell_row.items() if n < 30}
    check(f"full/{split}: every cell/row defect type has >= 30 samples", not thin, thin)
    check(f"full/{split}: every decoy type has >= 15 samples", all(n >= 15 for k, n in lc.items() if k.startswith("decoy:")),
          {k: n for k, n in lc.items() if k.startswith("decoy:") and n < 15})
    check(f"full/{split}: the ask-vs-fix path is exercised (>= 5 ambiguous date columns)", lc.get("defect:date_ambiguous", 0) >= 5, lc.get("defect:date_ambiguous"))
    check(f"full/{split}: file-level conditions each occur in >= 3 cases", all(lc.get(f"defect:{d}", 0) >= 3 for d in ("bom", "header_offset", "total_row")), {d: lc.get(f"defect:{d}", 0) for d in ("bom", "header_offset", "total_row")})

# ── manifest tripwire ────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmp:
    rc = cb.main(["build", "--profile", "mini", "--out", tmp])
    mpath = os.path.join(tmp, "MANIFEST.mini.json")
    check("build writes data files and a manifest", rc == 0 and os.path.exists(mpath) and os.path.exists(os.path.join(tmp, "dev", "dev-orders-120-us", "data.csv")))
    check("verify passes on untouched output", cb.main(["verify", "--profile", "mini", "--manifest", mpath]) == 0)
    m = json.load(open(mpath, encoding="utf-8"))
    m["splits"]["frozen"]["sha256"] = "0" * 64
    json.dump(m, open(mpath, "w", encoding="utf-8"))
    check("verify fails when a split's hash changes", cb.main(["verify", "--profile", "mini", "--manifest", mpath]) == 1)
    m["splits"]["frozen"]["sha256"] = cb.build("mini")["splits"]["frozen"]["sha256"]
    m["vocab_sha256"] = "0" * 64
    json.dump(m, open(mpath, "w", encoding="utf-8"))
    check("verify fails when the vocabulary changes", cb.main(["verify", "--profile", "mini", "--manifest", mpath]) == 1)

print()
if FAIL:
    print(f"{len(FAIL)} check(s) failed: {FAIL}")
    sys.exit(1)
print("all checks passed")
