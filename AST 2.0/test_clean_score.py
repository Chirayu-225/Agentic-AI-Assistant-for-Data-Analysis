"""
test_clean_score.py — does the scorer measure what it claims, and can it be gamed?
Checked with reference pipelines whose correct scores are known in advance (an oracle,
a do-nothing pipeline, one that flags everything, one that corrupts, one that silently
drops rows), plus the adapter's row-to-line mapping and the frozen-run safeguards.
Stdlib only; the final block exercises the real cleaner and runs only where pandas is installed.
Run: python test_clean_score.py
"""
import contextlib
import datetime as dt
import io
import json
import os
import sys
import tempfile

import clean_adapt as ca
import clean_score as cs

FAIL = []
def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {str(detail)[:300]}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)

# ── value comparison ─────────────────────────────────────────────────────────
check("numbers compare by value", cs.norm("money", "1234.5") == cs.norm("money", "1234.50") == cs.norm("money", " 1234.500 "))
check("formatted money is NOT silently parsed (leaving it unconverted must score as unconverted)", cs.norm("money", "$1,234.50") != cs.norm("money", "1234.50"))
check("a unit suffix is not parsed either", cs.norm("unit", "2.5 kg") != cs.norm("unit", "2.50"))
check("dates compare by ISO day, timestamps included", cs.norm("date", "2024-03-05 00:00:00") == cs.norm("date", "2024-03-05"))
check("a locale-formatted date is not the ISO date", cs.norm("date", "05.03.2024") != cs.norm("date", "2024-03-05"))
check("booleans compare case-insensitively but variants do not", cs.norm("bool", "True") == cs.norm("bool", "true") and cs.norm("bool", "Y") != cs.norm("bool", "true"))
check("text is exact: whitespace and case are real differences", cs.norm("text", " Neil ") != cs.norm("text", "Neil") and cs.norm("text", "NEIL") != cs.norm("text", "Neil"))
check("leading zeros are real differences in identifiers", cs.norm("idnum", "042425") != cs.norm("idnum", "42425"))
check("adapter: 1234.50 and 1234.5 are the same value", ca.same_value("1234.50", 1234.5))
check("adapter: an ID that lost its leading zero IS a change", not ca.same_value("042425", "42425"))
check("adapter: whitespace around text is a change", not ca.same_value(" x ", "x"))
check("adapter: whitespace around a NUMBER is a change too (a padded ID that was trimmed is a fix)", not ca.same_value("  13220", "13220") and not ca.same_value("13220", " 13220"))
check("adapter: a number that merely gained '.0' is still the same value", ca.same_value("13220", "13220.0"))
check("adapter: a boolean read as text and written back ('true' -> 'True') is not an edit", ca.same_value("true", "True") and not ca.same_value("true", "false"))

# ── adapter core: source coordinates survive dropped rows ────────────────────
RAW = b"id,name,amt\n1,a,10\n\n2,b,20\n3,c,30\n4,d,40\n"   # line 1 header, 2 rec, 3 BLANK, 4/5/6 rec
raw = ca.parse_raw(RAW)
check("parse_raw: header line, records and blank lines in source coordinates",
      raw["header_line"] == 1 and [l for l, _ in raw["records"]] == [2, 4, 5, 6] and raw["blank_lines"] == [3])
# a cleaner that drops the FIRST record and a middle one, re-indexes, and changes one cell
out_lines, out_rows = [4, 6], [["2", "b", "20"], ["4", "D", "40"]]
LOG2 = ["✓ Excluded 2 total/summary row(s) so they aren't counted as data"]
doc = ca.build_predictions("t", raw, ["id", "name", "amt"], ["id", "name", "amt"], out_rows, out_lines, LOG2, {"fix", "quarantine"})
P = doc["predictions"]
key = lambda p: (p["line"], p["col_index"], p["action"])
check("rows dropped early: later rows keep their SOURCE line numbers",
      sorted(map(key, P), key=lambda t: (t[0], t[1] or 0)) == [(2, None, "quarantine"), (3, None, "ignore"), (5, None, "quarantine"), (6, 2, "fix")], [key(p) for p in P])
check("the changed cell carries its new value, the untouched cells carry nothing", [p["value"] for p in P if p["action"] == "fix"] == ["D"])
check("a drop the log accounts for is 'quarantine'", all(p["action"] == "quarantine" for p in P if p["line"] in (2, 5)))
check("a blank line the reader skipped is 'ignore', not a drop", [p["action"] for p in P if p["line"] == 3] == ["ignore"])
d2 = ca.build_predictions("t", raw, ["id", "name", "amt"], ["id", "name", "amt"], out_rows, out_lines, [], {"fix"})
check("a drop with no log entry is 'dropped_unreported'", [p["action"] for p in d2["predictions"] if p["line"] in (2, 5)] == ["dropped_unreported"] * 2)
d3 = ca.build_predictions("t", raw, ["id", "name", "amt"], ["id", "name", "amt"], out_rows, out_lines, ["✓ Excluded 1 total/summary row(s)"], {"fix"})
check("a log that accounts for fewer rows than were removed is not trusted", [p["action"] for p in d3["predictions"] if p["line"] in (2, 5)] == ["dropped_unreported"] * 2)
d4 = ca.build_predictions("t", ca.parse_raw(b"id,zip\nA,042425\nB,1234.50\nC, x \n"), ["id", "zip"], ["id", "zip"],
                          [["A", "42425"], ["B", "1234.5"], ["C", "x"]], [2, 3, 4], [], {"fix"})
check("leading-zero loss and trimmed whitespace are visible; 1234.50 -> 1234.5 is not",
      sorted((p["line"], p["value"]) for p in d4["predictions"]) == [(2, "42425"), (4, "x")], d4["predictions"])
d5 = ca.build_predictions("t", ca.parse_raw(b"id,note,amt\nA,hello,5\n"), ["id", "note", "amt"], ["id", "amt", "amt_original"],
                          [["A", "5", "keep"]], [2], [], {"fix"})
check("columns are mapped by name (a dropped column does not shift the rest); side columns are ignored",
      d5["predictions"] == [] and d5["structure"]["columns"] == ["id", "amt"], d5)
d6 = ca.build_predictions("t", ca.parse_raw(b"id,v\nA,3\n"), ["id", "v"], ["id", "v"], [["A", ""]], [2], [], {"blank"})
check("a value that became empty is a 'blank'", d6["predictions"][0]["action"] == "blank" and d6["predictions"][0]["value"] is None)
check("a crash is reported as a failed case, with its capabilities", ca.failure("c", "boom", {"fix"})["status"] == "error")

# ── reference pipelines: known answers ───────────────────────────────────────
def run_ref(name, profile="mini", split="dev"):
    cases = cs.load_cases(profile, split)
    return cases, cs.score(cases, cs.make_pipeline(name, cases))

def totals(r):
    d = r["by_defect"]
    return {k: sum(v[k] for v in d.values()) for k in ("n", "scored", "detected", "correct", "wrong", "silent", "missed", "unsupported")}

for split in ("dev", "frozen"):
    cases, r = run_ref("oracle", split=split)
    t = totals(r)
    check(f"{split}/oracle: every defect handled correctly", t["correct"] == t["scored"] > 20, t)
    check(f"{split}/oracle: touches no decoy, damages nothing", all(v["hits"] == 0 for v in r["by_decoy"].values())
          and r["damage"]["cells_corrupted"] == 0 and r["damage"]["rows_wrongly_removed_rate"] == 0)
    check(f"{split}/oracle: precision is 100%", r["precision"]["precision"] == 1.0, r["precision"])
    check(f"{split}/oracle: never asks where a fix was right", r["ask_vs_fix"]["over_ask"] == 0)
    check(f"{split}/oracle: improves the cells (some defects need a human, so not 100%)",
          r["end_state"]["correct_after"] > r["end_state"]["correct_before"])
    _, n = run_ref("null", split=split)
    tn = {k: sum(v[k] for d, v in n["by_defect"].items() if d not in cs.FILE_LEVEL)
          for k in ("detected", "correct", "wrong", "silent")}   # file-level conditions are judged on the resulting structure
    check(f"{split}/null: detects nothing, damages nothing, touches no decoy",
          tn["detected"] == 0 and tn["correct"] == tn["wrong"] == tn["silent"] == 0 and all(v["hits"] == 0 for v in n["by_decoy"].values())
          and n["damage"]["cells_corrupted"] == 0 and n["end_state"]["correct_after"] == n["end_state"]["correct_before"])
    check(f"{split}/null: structural handling that needs no action (BOM, trailing delimiter) can still be correct, a header offset cannot",
          n["by_defect"].get("header_offset", {"correct": 0})["correct"] == 0)

# flagging everything must NOT look good
cases, f = run_ref("flag_all")
tf = totals(f)
check("flag-everything: high detection ...", tf["detected"] / tf["scored"] > 0.6, tf)
check("... but low correctness (a flag is the right answer only where a flag is accepted)", tf["correct"] / tf["scored"] < 0.3, tf)
check("flag-everything: hits every decoy", all(v["hit_rate"] == 1.0 for v in f["by_decoy"].values()), f["by_decoy"])
check("flag-everything: touches every cell of the control files", f["controls"]["cells_touched_rate"] == 1.0)
check("flag-everything: precision is low", f["precision"]["precision"] < 0.5, f["precision"])
_, c = run_ref("corrupter")
check("a pipeline that corrupts values shows heavy cell damage", c["damage"]["cells_corrupted_rate"] > 0.8, c["damage"])
check("... and scores no correct fixes", totals(c)["correct"] == 0)
_, dr = run_ref("dropper")
check("a pipeline that silently drops 1 row in 10 shows it as wrongly removed rows", 0.05 < dr["damage"]["rows_wrongly_removed_rate"] < 0.15, dr["damage"])
check("... all of them silent", dr["damage"]["rows_wrongly_removed_reported"] == 0 and dr["damage"]["rows_wrongly_removed_silent"] > 0)

# ── reformatting a defective cell is not an attempt at the defect ────────────
cov = cs.load_cases("coverage", "dev")
cov_oracle = cs.make_pipeline("oracle", cov)
def reformat_only(case_id, data, comp):
    case = next(c for c in cov if c.id == case_id)
    d = cov_oracle(case_id, data, comp)
    targets = {(L["line"], c["col_index"]): c["unfixed"] for L in case.labels for c in L["cells"]
               if L["defect"] in ("outlier_magnitude", "future_date", "end_before_start") and c["unfixed"] is not None}
    d["predictions"] = [p for p in d["predictions"] if (p["line"], p["col_index"]) not in targets and not (
        p["action"] in ("flag", "quarantine") and any(p["line"] == ln for ln, _ in targets))]
    d["predictions"] += [cs._p(ln, ci, "fix", u) for (ln, ci), u in targets.items()]
    return d
rr = cs.score(cov, reformat_only)
for dname in ("outlier_magnitude", "future_date", "end_before_start"):
    v = rr["by_defect"][dname]
    check(f"{dname}: only converting the cell's format is 'missed', not 'wrong' and not 'detected'",
          v["wrong"] == 0 and v["detected"] == 0 and v["missed"] == v["n"] > 0, v)
check("... while really fixing it still counts", cs.score(cov, cov_oracle)["by_defect"]["outlier_magnitude"]["correct"] > 0)

# ── capabilities: 'cannot do' is not 'missed' ────────────────────────────────
cases = cs.load_cases("mini", "dev")
oracle = cs.make_pipeline("oracle", cases)
def no_ask(case_id, data, comp):
    d = oracle(case_id, data, comp)
    d["capabilities"] = sorted(set(d["capabilities"]) - {"ask"})
    d["predictions"] = [p for p in d["predictions"] if p["action"] != "ask"]
    return d
r = cs.score(cases, no_ask)
da = r["by_defect"]["date_ambiguous"]
check("a pipeline that cannot ask has the ambiguous dates reported as unsupported, not as misses",
      da["unsupported"] == da["n"] > 0 and da["scored"] == 0 and da["missed"] == 0 and da["correct_rate"] is None, da)
check("ask-vs-fix reports them as unsupported", r["ask_vs_fix"]["unsupported"] == r["ask_vs_fix"]["ambiguous_columns"] > 0, r["ask_vs_fix"])
rn = cs.score(cases, cs.make_pipeline("null", cases))
check("the same labels ARE misses for a pipeline that could have asked", rn["by_defect"]["date_ambiguous"]["missed"] == da["n"] and rn["by_defect"]["date_ambiguous"]["unsupported"] == 0)

def autofix(case_id, data, comp):
    case = next(c for c in cases if c.id == case_id)
    d = oracle(case_id, data, comp)
    d["predictions"] = [p for p in d["predictions"] if p["action"] != "ask"]
    for L in case.labels:
        if L["defect"] == "date_ambiguous":
            ci = L["cells"][0]["col_index"]
            d["predictions"] += [cs._p(ln, ci, "fix", case.truth[ln][ci - 1]) for ln in case.truth if case.truth[ln][ci - 1]]
    return d
ra = cs.score(cases, autofix)
check("guessing the day/month order is 'autofixed' and wrong, even when the guess matches the truth",
      ra["by_defect"]["date_ambiguous"]["wrong"] == ra["by_defect"]["date_ambiguous"]["n"] and ra["ask_vs_fix"]["autofixed"] == ra["ask_vs_fix"]["ambiguous_columns"] > 0, ra["ask_vs_fix"])

def over_ask(case_id, data, comp):
    case = next(c for c in cases if c.id == case_id)
    d = oracle(case_id, data, comp)
    d["predictions"] += [cs._p(L["line"], L["cells"][0]["col_index"], "ask") for L in case.labels
                         if L["kind"] == "defect" and L["defect"] == "whitespace"]
    return d
ro = cs.score(cases, over_ask)
check("asking where a fix was right is counted as over-ask", ro["ask_vs_fix"]["over_ask"] > 0 and ro["ask_vs_fix"]["over_ask_rate"] > 0, ro["ask_vs_fix"])

# ── duplicates: one surviving copy is the right answer, whichever copy ───────
def dup_pipeline(which, action="quarantine"):
    def run(case_id, data, comp):
        case = next(c for c in cases if c.id == case_id)
        d = oracle(case_id, data, comp)
        d["predictions"] = [p for p in d["predictions"] if not (p["action"] == "quarantine" and p["line"] in
                            {L["line"] for L in case.labels if L["defect"] in ("dup_exact", "dup_near")})]
        for L in case.labels:
            if L["defect"] in ("dup_exact", "dup_near"):
                lines = {"extra": [L["line"]], "original": [L["related_line"]], "both": [L["line"], L["related_line"]]}[which]
                d["predictions"] += [cs._p(ln, None, action) for ln in lines]
        return d
    return run
dd = lambda r_: r_["by_defect"]["dup_exact"]
r1, r2, r3 = (cs.score(cases, dup_pipeline(w)) for w in ("extra", "original", "both"))
check("removing the EXTRA copy of a duplicate is correct", dd(r1)["correct"] == dd(r1)["n"] > 0)
check("removing the ORIGINAL instead is equally correct (one copy survives)", dd(r2)["correct"] == dd(r2)["n"])
check("removing the original does not count as damage to a clean row", r2["damage"]["rows_wrongly_removed_rate"] == 0, r2["damage"])
check("removing BOTH copies is wrong", dd(r3)["wrong"] == dd(r3)["n"] and dd(r3)["correct"] == 0)
r4 = cs.score(cases, dup_pipeline("extra", "dropped_unreported"))
check("removing a duplicate without reporting it is 'silent', not correct", dd(r4)["silent"] == dd(r4)["n"] and dd(r4)["correct"] == 0)

# ── denominators ─────────────────────────────────────────────────────────────
def remove_clean(k):
    def run(case_id, data, comp):
        case = next(c for c in cases if c.id == case_id)
        d = oracle(case_id, data, comp)
        bad = {L["line"] for L in case.labels if L["kind"] == "defect" and L["scope"] in ("cell", "row") and L["line"]}
        bad |= {L["related_line"] for L in case.labels if L["defect"] in ("dup_exact", "dup_near")}
        clean = [ln for ln in sorted(case.truth) if ln not in bad]
        d["predictions"] += [cs._p(ln, None, "quarantine") for ln in clean[:k]]
        return d
    return run
r5 = cs.score(cases, remove_clean(4))
expected_clean = 0
for case in cases:
    bad = {L["line"] for L in case.labels if L["kind"] == "defect" and L["scope"] in ("cell", "row") and L["line"]}
    bad |= {L["related_line"] for L in case.labels if L["defect"] in ("dup_exact", "dup_near")}
    expected_clean += len(set(case.truth) - bad)
check("'wrongly removed' is measured over rows that are clean or decoys (denominator computed independently)",
      r5["damage"]["clean_or_decoy_rows"] == expected_clean and r5["damage"]["rows_wrongly_removed_reported"] == 4 * len(cases)
      and r5["damage"]["rows_wrongly_removed_rate"] == round(4 * len(cases) / expected_clean, 4), r5["damage"])

# a drop early in the file, scored end to end, must hit exactly that row and nothing after it
case = next(c for c in cases if c.spec.family == "orders")
records = ca.parse_raw(case.data)
first = records["records"][0][0]
keep = [(ln, f) for ln, f in records["records"] if ln != first]
def early_drop(case_id, data, comp):
    raw_ = ca.parse_raw(data)
    c_ = next(x for x in cases if x.id == case_id)
    if case_id != case.id:
        return cs._doc(case_id, [], {"header_line": 1, "columns": list(c_.names)})
    return ca.build_predictions(case_id, raw_, raw_["header"], raw_["header"], [f for _, f in keep], [ln for ln, _ in keep],
                                [f"✓ Excluded 1 total/summary row(s)"], cs.ALL_CAPS)
pi = cs.PI(early_drop(case.id, case.data, {}), case)
check("an early dropped row scores as exactly that one source line, with nothing after it shifted", set(pi.removed) == {first}, set(pi.removed))
re_ = cs.score(cases, early_drop)
check("... and no cell anywhere is reported as changed", re_["damage"]["cells_corrupted"] == 0 and re_["precision"]["interventions"] == 1, re_["precision"])

# ── frozen safeguards ────────────────────────────────────────────────────────
NOW = dt.datetime(2026, 11, 5, 12, 0, 0)
def frozen(adapter, commit, dirty=0, now=NOW, log=None, manifest=True):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cs.main(["--adapter", adapter, "--frozen", "--profile", "mini"], git=lambda: (commit, dirty), log_path=log,
                     now=now, manifest_check=lambda: manifest)
    return rc, buf.getvalue()

with tempfile.TemporaryDirectory() as tmp:
    log = os.path.join(tmp, "FROZEN_RUNS.jsonl")
    rc, out = frozen("null", "aaa111", dirty=3, log=log)
    check("frozen refuses to run on a dirty tree", rc == 2 and "REFUSED" in out and not os.path.exists(log), out[:200])
    rc, out = frozen("null", None, log=log)
    check("frozen refuses when git state is unavailable", rc == 2 and "REFUSED" in out)
    rc, out = frozen("null", "aaa111", log=log, manifest=False)
    check("frozen refuses when the data no longer matches the manifest", rc == 2 and "MANIFEST" in out and not os.path.exists(log))
    rc, out = frozen("null", "aaa111", log=log)
    first_line = out.splitlines()[0]
    check("the run count and last run are printed BEFORE the score", rc == 0 and first_line.startswith("FROZEN RUN #1") and "last run: never" in first_line
          and out.index("FROZEN RUN #1") < out.index("defect type"), first_line)
    recs = [json.loads(l) for l in open(log, encoding="utf-8")]
    check("the run is appended to the log with its commit and a scores digest", len(recs) == 1 and recs[0]["commit"] == "aaa111" and len(recs[0]["digest"]) == 12 and recs[0]["adapter"] == "null")
    rc, out = frozen("null", "aaa111", log=log, now=NOW + dt.timedelta(minutes=10))
    check("a second run within the hour is warned about, and numbered", "FROZEN RUN #2" in out and "under an hour ago" in out, out[:300])
    rc, out = frozen("oracle", "bbb222", log=log, now=NOW + dt.timedelta(hours=3))
    check("a run on a new commit whose score differs from the last is not yet a pattern", "different commits and scored differently" not in out, out[:400])
    rc, out = frozen("null", "ccc333", log=log, now=NOW + dt.timedelta(hours=5))
    check("two consecutive runs on different commits with different scores trigger the tuning warning", "signature of tuning" in out, out[:500])
    for i in range(3):
        rc, out = frozen("null", "ccc333", log=log, now=NOW + dt.timedelta(hours=6 + i))
    check("five runs in a week trigger the frequency warning", "frozen runs in 7 days" in out, out[:400])
    check("the log is append-only in practice: every run is a line", len(open(log, encoding="utf-8").readlines()) == 7)
    rc, out = frozen("crash", "ddd444", log=log, now=NOW + dt.timedelta(days=30))
    check("frozen never shows per-case failures or case ids", rc == 0 and "boom" not in out and "frozen-orders" not in out and "frozen-staff" not in out and "crashed cases" not in out, out[-400:])
    check("... though the aggregate crash count is visible", "crashed" in out)

devbuf = io.StringIO()
with contextlib.redirect_stdout(devbuf):
    cs.main(["--adapter", "crash", "--profile", "mini"])
check("dev shows crashed cases, with their ids and errors, so they can be fixed", "crashed cases" in devbuf.getvalue() and "boom" in devbuf.getvalue() and "dev-orders" in devbuf.getvalue())
rcr = cs.score(cs.load_cases("mini", "dev"), cs.make_pipeline("crash", cs.load_cases("mini", "dev")))
check("labels in crashed cases count as misses and are also shown separately",
      all(v["correct"] == 0 and v["case_failed"] > 0 for k, v in rcr["by_defect"].items() if k not in cs.FILE_LEVEL and v["unsupported"] == 0), rcr["by_defect"])

# ── the real cleaner (only where pandas is installed) ────────────────────────
try:
    import pandas  # noqa: F401
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False
    print("[SKIP] real-cleaner integration: pandas is not installed here (runs on your machine)")
if HAVE_PANDAS:
    import clean_adapter_legacy as la
    tiny = b"id,name,amount\nA1,x,10\nTOTAL,,10\nA2,y,20\nA3,z,30\n"
    d = la.run("tiny", tiny, {})
    check("legacy: runs on a small file", d["status"] == "ok", d.get("error"))
    rm = {p["line"]: p["action"] for p in d.get("predictions", []) if p["col_index"] is None}
    check("legacy: the TOTAL row is reported as quarantined at ITS source line (3), and nothing after it moves", rm == {3: "quarantine"}, d.get("predictions"))
    check("legacy: it declares what it cannot say", "ask" not in d["capabilities"] and "flag" not in d["capabilities"])
    hdr = b"Data export\nGenerated by ERP\nid,name\nA,b\n"
    check("legacy: a file with title lines above the header fails and is reported as a failed case", la.run("hdr", hdr, {})["status"] == "error")
    cases_ = cs.load_cases("mini", "dev")
    ctrl = next(c for c in cases_ if c.control)
    dc = la.run(ctrl.id, ctrl.data, ctrl.companions)
    check("legacy: runs on a benchmark control file", dc["status"] == "ok", dc.get("error"))
    rr = cs.score(cases_, la.run)
    check("legacy: scores end to end (no exception, labels accounted for)", rr["cases"] == 3 and sum(v["n"] for v in rr["by_defect"].values()) > 0)

print()
if FAIL:
    print(f"{len(FAIL)} check(s) failed: {FAIL}")
    sys.exit(1)
print("all checks passed")
