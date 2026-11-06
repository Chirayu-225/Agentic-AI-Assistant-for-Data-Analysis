"""
test_exp.py — the experiment's question set, harness, guards and decision logic. No model is called: the harness is
exercised with the mechanical solver and with fake solvers, and the decision code with synthetic results whose correct
verdict is known in advance. Run: python test_exp.py
"""
import contextlib
import csv
import io
import json
import os
import sys
import tempfile
from collections import Counter

import exp_analyze as ea
import exp_questions as eq
import exp_rules
import exp_run as er

FAIL = []
def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {str(detail)[:300]}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)

cases, questions = eq.build_exp()
qmap = {q["id"]: q for q in questions}

# ── the question set ─────────────────────────────────────────────────────────
check("50 questions: 40 single-defect, 10 compound", len(questions) == 50 and Counter(q["tier"] for q in questions) == {"single": 40, "compound": 10}, Counter(q["tier"] for q in questions))
check("question ids are unique", len(qmap) == 50)
check("the set is deterministic (same hash twice)", eq.questions_sha256(questions) == eq.questions_sha256(eq.build_exp()[1]) and eq.data_sha256() == eq.data_sha256())
check("structural vs value-level kinds are as registered", {q["tag"] for q in questions if q["kind"] == "structural"} == set(exp_rules.RULES["structural_tags"]))
check("every expected answer is a number", all(q["answer_type"] == "number" and float(q["expected"]) == float(q["expected"]) for q in questions))
leak = [q["id"] for q in questions if any(w in q["question"].lower() for w in ("dirty", "clean", "sentinel", "footer", "duplicate", "outlier", "defect", "missing", "n/a"))]
check("no question hints at the defect it is testing", not leak, leak)

# each table carries exactly the defect(s) it is meant to carry (single tier: one type)
EXPECT = {"footer": {"footer_note"}, "total": {"total_row"}, "header": {"header_offset"}, "ragged": {"ragged_row"},
          "bomtrail": {"bom", "trailing_delimiter"}, "dupexact": {"dup_exact"}, "dupnear": {"dup_near"}, "sentinel": {"sentinel_null"},
          "numtext": {"number_as_text"}, "outlier": {"outlier_magnitude"}, "impossible": {"impossible_value"}, "future": {"future_date"},
          "datemix": {"date_mixed_format"}, "catvar": {"category_variant"}, "whitespace": {"whitespace"}, "swap": {"swapped_values"},
          "unit": {"unit_mismatch"}, "pct": {"percent_scale"}, "zeros": {"lost_leading_zero"}, "endstart": {"end_before_start"}, "orphan": {"orphan_fk"}}
bad = []
for cid, (spec, files, doc) in cases.items():
    tag = spec.flags["tag"]
    got = {L["defect"] for L in doc["labels"] if L["kind"] == "defect"}
    if tag in EXPECT and got != EXPECT[tag]:
        bad.append((tag, got))
    if tag not in EXPECT and len(got - {"total_row", "footer_note"}) < 2:
        bad.append((tag, "compound table has fewer than 2 defect types", got))
check("single-defect tables carry exactly their defect; compound tables carry 2-3", not bad, bad[:3])
check("the ordinary join exists only where the question needs it", [q["id"] for q in questions if q["join"]] == ["orphan-1", "orphan-2"])
check("only the orphan table has a customers file with a tier", [c for c, (_, f, _) in cases.items() if "customers.csv" in f and b"tier" in f["customers.csv"].split(b"\n")[0]] == [c for c in cases if cases[c][0].flags["tag"] == "orphan"])

# ── the ceiling: attainable-clean ────────────────────────────────────────────
diffs = Counter()
for cid, (spec, files, doc) in cases.items():
    full = {int(r["__line"]): r for r in csv.DictReader(io.StringIO(files["truth.csv"].decode()))}
    att = {int(r["__line"]): r for r in eq.attainable_rows(files, doc)}
    for ln in full:
        for k in full[ln]:
            if full[ln][k] != att[ln][k]:
                diffs[spec.flags["tag"]] += 1
                assert att[ln][k] == "", "attainable only ever BLANKS a cell, it never invents one"
unrec = {t for t in diffs}
check("the ceiling blanks only cells of unrecoverable defects (and short ragged rows' missing fields)",
      unrec <= {"sentinel", "impossible", "future", "endstart", "orphan", "ragged", "c1", "c3", "c4"}, unrec)
check("recoverable defects leave the ceiling identical to the full truth", not ({"numtext", "catvar", "whitespace", "datemix", "swap", "unit", "pct", "zeros", "dupexact", "dupnear", "footer", "total"} & unrec))
check("information-loss is flagged on exactly the questions whose answer moves", {q["id"] for q in questions if q["info_lost"]} ==
      {q["id"] for q in questions if q["expected"] != q["expected_true"]} and any(q["info_lost"] for q in questions))
check("blanking a sentinel's cell moves the question it feeds: sentinel-2 attainable differs from the full truth", qmap["sentinel-2"]["info_lost"])
check("an unrelated defect does not: numtext keeps both answers equal", not qmap["numtext-1"]["info_lost"])

# ── independent ground truth: stdlib truth vs pandas on the clean arm ─────────
res = []
for q in questions:
    files, doc = cases[q["case_id"]][1], cases[q["case_id"]][2]
    data = eq.attainable_csv(files, doc)
    r = er.mechanical_solver(data, q, files.get("customers.csv"), "t")
    res.append((q["id"], er.classify(q, r)[0]))
check("pandas on the clean arm reproduces the stdlib ground truth on all 50 questions (two implementations agree)",
      all(o == "correct" for _, o in res), [x for x in res if x[1] != "correct"])

# ── the arms ─────────────────────────────────────────────────────────────────
spec, files, doc = cases[next(c for c in cases if c.endswith("-header"))]
d_clean, e1, _, _ = er.prepare("clean", files, doc)
d_dirty, e2, _, _ = er.prepare("dirty", files, doc)
d_s0, e3, _, mb = er.prepare("stage0", files, doc)
check("arms: clean = the ceiling table, dirty = the file as uploaded, stage0 = a table", d_clean == eq.attainable_csv(files, doc) and d_dirty == files["data.csv"] and d_s0 and not (e1 or e2 or e3))
check("arms: stage0 turns a file with title lines into a readable table (the old reader cannot)",
      er.mechanical_solver(d_dirty, qmap["header-1"], None, "t")["success"] is False and er.mechanical_solver(d_s0, qmap["header-1"], None, "t")["success"] is True)
check("arms: preparation time and peak memory are recorded per arm", isinstance(mb, float))

# ── outcomes ─────────────────────────────────────────────────────────────────
q0 = qmap["footer-1"]
cl = lambda **r: er.classify(q0, {"success": r.get("success", True), "resultType": "scalar", "resultData": r.get("v"), "error": r.get("error")})[0]
check("outcome: right / wrong / crash / infra are told apart",
      cl(v=601) == "correct" and cl(v=604) == "wrong" and cl(success=False, error="Could not parse file: boom") == "crash"
      and cl(success=False, error="ARM_CRASH: unreadable") == "crash"
      and cl(success=False, error="The AI provider's daily usage quota has been used up, so this can't be answered right now.") == "infra_error"
      and cl(success=False, error="Analysis pipeline failed: LLM code call failed: Error code: 429 Rate limit reached") == "infra_error"
      and cl(success=False, error="Result shaping failed") == "error")

# ── a mechanical pass: plumbing, resume, budget cap ──────────────────────────
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "p.jsonl")
    out = []
    s1 = er.run_pass("mechanical", "t", "single", 1, 0, path, questions, cases, out=out.append)
    recs = [json.loads(l) for l in open(path)]
    check("a full single-tier pass runs 40 questions x 3 arms", s1["runs"] == 120 and len(recs) == 120)
    check("the clean arm answers every question correctly (the ceiling is attainable and the questions are sound)",
          all(r["outcome"] == "correct" for r in recs if r["arm"] == "clean"))
    check("every record carries the fields the analysis needs", all({"qid", "tag", "tier", "kind", "arm", "repeat", "outcome", "actual", "expected", "calls", "tokens_prompt", "prep_peak_mb", "solver"} <= set(r) for r in recs))
    s2 = er.run_pass("mechanical", "t", "single", 1, 0, path, questions, cases, out=out.append)
    check("re-running the same pass resumes and does no work", s2["runs"] == 0 and len(open(path).readlines()) == 120)
    p2 = os.path.join(tmp, "q.jsonl")
    s3 = er.run_pass("mechanical", "t", "single", 1, 7, p2, questions, cases, out=out.append)
    s4 = er.run_pass("mechanical", "t", "single", 1, 7, p2, questions, cases, out=out.append)
    blocks_ok = all(len({r["arm"] for r in rs}) == 3 for rs in (lambda d: [[r for r in d if r["qid"] == q] for q in {r["qid"] for r in d}])([json.loads(l) for l in open(p2)]))
    check("--max-runs is checked BETWEEN question blocks (never mid-block), and the next call resumes where it stopped",
          s3["runs"] == 9 and s3["stopped"] == "max_runs" and s4["runs"] == 9 and len({(r["qid"], r["arm"]) for r in map(json.loads, open(p2))}) == 18 and blocks_ok, (s3["runs"], s4["runs"]))
    first = [json.loads(l) for l in open(p2)][:3]
    check("all arms of a question run together (blocking by question)", {r["qid"] for r in first} == {first[0]["qid"]} and [r["arm"] for r in first] == ["clean", "dirty", "stage0"])

# infra errors are excluded and marked, never substituted; a daily quota stops the run
calls = []
def flaky(data, q, customers, session):
    calls.append(q["id"])
    if len(calls) == 2:
        return {"success": False, "resultType": "error", "resultData": None, "error": "Error code: 429 - Rate limit reached for model on tokens per minute"}
    if len(calls) == 5:
        return {"success": False, "resultType": "error", "resultData": None, "error": "The AI provider's daily usage quota has been used up, so this can't be answered right now."}
    return er.mechanical_solver(data, q, customers, session)
orig = er.SOLVERS["mechanical"]
er.SOLVERS["mechanical"] = flaky
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "f.jsonl")
    s = er.run_pass("mechanical", "t", "single", 1, 0, path, questions, cases, out=lambda *_: None)
    recs = [json.loads(l) for l in open(path)]
er.SOLVERS["mechanical"] = orig
check("a short rate limit is recorded as infra_error and the run carries on (no substitution)", recs[1]["outcome"] == "infra_error" and recs[2]["outcome"] != "infra_error")
check("a daily quota stops the whole run, with the reason", s["stopped"] == "daily_quota" and len(recs) == 5, (s["stopped"], len(recs)))
maj, infra = ea.collapse(recs)
check("infra errors are excluded from every rate and counted per cell", (recs[1]["qid"], recs[1]["arm"]) not in maj and infra[(recs[1]["qid"], recs[1]["arm"])] == 1)

# ── guards on an LLM run ─────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmp:
    old = (er.PREREG, er.RUNS_LOG)
    er.PREREG, er.RUNS_LOG = os.path.join(tmp, "PREREG.json"), os.path.join(tmp, "RUNS.jsonl")
    clean_git = lambda: ("abc12345", 0)
    ok, lines, _ = er.guard_llm(questions, git=clean_git)
    check("an LLM run refuses without a pre-registration", not ok and "PREREG.json does not exist" in lines[0])
    check("--write-prereg writes it, and the harness then accepts a clean tree", er.write_prereg() == 0 and er.guard_llm(questions, git=clean_git)[0])
    ok, lines, _ = er.guard_llm(questions, git=lambda: ("abc12345", 3))
    check("an LLM run refuses on a dirty tree (a previous run's log line counts)", not ok and "uncommitted" in lines[0])
    check("an LLM run refuses when git state is unreadable", not er.guard_llm(questions, git=lambda: (None, None))[0])
    tampered = [dict(q) for q in questions]
    tampered[0]["expected"] = "1"
    ok, lines, _ = er.guard_llm(tampered, git=clean_git)
    check("editing a question after pre-registration is caught", not ok and "question set no longer matches" in lines[0])
    pre = json.load(open(er.PREREG))
    pre["model"]["code_model"] = "some-other-model"
    json.dump(pre, open(er.PREREG, "w"))
    ok, lines, _ = er.guard_llm(questions, git=clean_git)
    check("a different model than the pinned one is refused", not ok and "model" in lines[0].lower())
    pre["model"] = er.model_pin()
    pre["rules"]["headroom"]["min_points"] = 5.0
    json.dump(pre, open(er.PREREG, "w"))
    check("changing a rule after the fact cannot be hidden: write_prereg refuses to overwrite a differing pre-registration", er.write_prereg() == 2)
    check("the guard prints the run number and last run first", er.guard_llm(questions, git=clean_git)[1][0].startswith("EXPERIMENT RUN #1"))
    with open(er.RUNS_LOG, "w") as f:
        f.write(json.dumps({"ts": "2026-11-05T13:19:07", "commit": "65054ac3", "pass": "full", "runs": 69, "stopped": "daily_quota"}) + "\n")
        f.write(json.dumps({"ts": "2026-11-06T09:00:00", "attest": True, "pass": "full", "system_sha": "x"}) + "\n")
    ok, lines, _ = er.guard_llm(questions, git=clean_git)
    check("an attestation line in RUNS.jsonl is not mistaken for a run (it has no commit, and must not crash the guard)",
          ok and lines[0].startswith("EXPERIMENT RUN #2") and "65054ac3" in lines[0], lines)
    er.PREREG, er.RUNS_LOG = old

def rec(qid, arm, rep, outcome, note=False, tier="single", kind="value", tag="x"):
    return {"qid": qid, "arm": arm, "repeat": rep, "outcome": outcome, "note": note, "tier": tier, "kind": kind, "tag": tag, "actual": "v", "solver": "llm",
            "pass": "full", "calls": 1, "tokens_prompt": 10, "tokens_completion": 5}

# ── a refused cell is retried, never left as a hole; pace mode waits out the provider's refilling quota ──
QUOTA = {"success": False, "resultType": "error", "resultData": None,
         "error": "The AI provider's daily usage quota has been used up, so this can't be answered right now."}
one_q = [q for q in questions if q["tag"] == "footer"][:1]
def scripted(plan):
    state = {"n": 0}
    def solver(data, q, customers, session):
        i = state["n"]
        state["n"] += 1
        step = plan[i] if i < len(plan) else "ok"
        if step == "quota":
            return dict(QUOTA)
        if step == "interrupt":
            raise KeyboardInterrupt
        return er.mechanical_solver(data, q, customers, session)
    return solver
def drive(plan, pace, max_attempts=None):
    sleeps, orig_sleep, orig_wait, orig_att = [], er.SLEEP, er.provider_wait, er.MAX_PACE_ATTEMPTS
    er.SLEEP, er.provider_wait = sleeps.append, lambda rec: 7.0
    if max_attempts is not None:
        er.MAX_PACE_ATTEMPTS = max_attempts
    orig = er.SOLVERS["mechanical"]
    er.SOLVERS["mechanical"] = scripted(plan)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "r.jsonl")
        summ = er.run_pass("mechanical", "t", "single", 1, 0, path, one_q, cases, out=lambda *_: None, pace=pace)
        recs = [json.loads(l) for l in open(path)]
        again = None
        if not pace:
            er.SOLVERS["mechanical"] = scripted([])          # a healthy provider on the second invocation
            again = er.run_pass("mechanical", "t", "single", 1, 0, path, one_q, cases, out=lambda *_: None)
            recs = [json.loads(l) for l in open(path)]
    er.SOLVERS["mechanical"], er.SLEEP, er.provider_wait, er.MAX_PACE_ATTEMPTS = orig, orig_sleep, orig_wait, orig_att
    return summ, recs, sleeps, again

summ, recs, _, again = drive(["ok", "quota"], pace=False)
check("without --pace a quota error stops the run after recording it", summ["stopped"] == "daily_quota" and [r["outcome"] for r in recs[:2]] == ["correct", "infra_error"])
check("on resume the unfinished BLOCK is re-run in full (a question's arms never straddle a pause), refused cell included",
      again["runs"] == 3 and [(r["arm"], r["outcome"] == "infra_error") for r in recs] == [("clean", False), ("dirty", True), ("clean", False), ("dirty", False), ("stage0", False)], [(r["arm"], r["outcome"]) for r in recs])
m, inf = ea.collapse(recs)
check("a cell that was run again counts ONCE (the latest record wins), and the refused attempt stays in the file as infra",
      inf[(one_q[0]["id"], "dirty")] == 1 and m[(one_q[0]["id"], "clean")]["n"] == 1 and m[(one_q[0]["id"], "dirty")]["n"] == 1)
summ, recs, sleeps, _ = drive(["ok", "quota", "quota"], pace=True)
check("--pace waits out the provider and retries the SAME run until it is served (no substitution, no stop)",
      summ["stopped"] is None and sleeps == [7.0, 7.0] and [(r["arm"], r["outcome"]) for r in recs][:4] == [("clean", "correct"), ("dirty", "infra_error"), ("dirty", "infra_error"), ("dirty", recs[3]["outcome"])] and len(recs) == 5, (sleeps, [(r["arm"], r["outcome"]) for r in recs]))
summ, recs, sleeps, _ = drive(["ok"] + ["quota"] * 20, pace=True, max_attempts=2)
check("--pace is bounded: it gives up on a run that is refused over and over, and says so", summ["stopped"] == "gave_up" and len(sleeps) == 2, (summ["stopped"], sleeps))
summ, recs, _, _ = drive(["ok", "interrupt"], pace=False)
check("Ctrl-C stops cleanly: finished runs are kept and the stop reason is reported (so the log line is still written)", summ["stopped"] == "interrupted" and len(recs) >= 1 and recs[0]["outcome"] == "correct", summ)
import llm_provider as lp
lp._tls.last_rate_limit = {"retry_after": 532.0, "daily": True}
check("the wait is the provider's own retry-after plus a margin", er.provider_wait({"detail": "x"}) == 547.0)
lp._tls.last_rate_limit = {"retry_after": 99999.0, "daily": True}
check("... capped at an hour, so a pathological value cannot hang the run for days", er.provider_wait({"detail": "x"}) == 3600.0)
lp.clear_last_rate_limit()
check("with no retry-after it falls back to conservative defaults", er.provider_wait({"detail": "The daily AI usage limit for this demo"}) == 900.0 and er.provider_wait({"detail": "429"}) == 120.0)
rep = ea.report([dict(rec("a1", "clean", 1, "correct"), tag="x", calls=1), dict(rec("a1", "dirty", 1, "correct"), tag="x", calls=0),
                 dict(rec("a2", "dirty", 1, "crash"), tag="x", calls=0)], [{"id": "a1", "tag": "x", "tier": "single", "kind": "value", "expected": "1", "question": "?"},
                 {"id": "a2", "tag": "x", "tier": "single", "kind": "value", "expected": "1", "question": "?"}])
check("the report names the questions the model never saw (answered by the rule-based shortcut), but not crashes", "WITHOUT the model" in rep and "a1" in rep.split("WITHOUT the model")[1].split("\n")[0] and "a2" not in rep.split("WITHOUT the model")[1].split("\n")[0])

# ── the system under test is frozen: stamps, fingerprint, continuity, attestation ──
import shutil
files = er.system_files()
check("the system under test is found by reading imports (the analysis path, the join path, the cleaner, Stage 0, the provider)",
      {"analysis_service.py", "llm_provider.py", "clean_read.py", "cleaning.py", "file_reader.py", "multi_table_service.py", "clean_adapter_stage0.py"} <= set(files), files)
check("... and it excludes the harness, the tests, the benchmark and the API (deployment work cannot trip the guard)",
      not ({"exp_run.py", "exp_analyze.py", "test_exp.py", "clean_bench.py", "clean_score.py", "api.py"} & set(files)), files)
with tempfile.TemporaryDirectory() as tmp:
    for name, body in (("analysis_service.py", "import helper\n"), ("helper.py", "X = 1\n"), ("unrelated.py", "Y = 1\n")):
        open(os.path.join(tmp, name), "w").write(body)
    f0 = er.fingerprint(tmp)
    open(os.path.join(tmp, "unrelated.py"), "w").write("Y = 2\n")
    f1 = er.fingerprint(tmp)
    open(os.path.join(tmp, "helper.py"), "w").write("X = 2\n")
    f2 = er.fingerprint(tmp)
check("editing a file the experiment imports changes the fingerprint; editing an unrelated file does not", f0 == f1 and f1 != f2)
stamp = {"commit": "abc", "model": "m", "temperature": 0.1, "system_sha": "s1"}
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "s.jsonl")
    er.run_pass("mechanical", "t", "single", 1, 0, path, one_q, cases, out=lambda *_: None, stamp=stamp)
    recs = [json.loads(l) for l in open(path)]
    check("every cell carries its commit, model, temperature and system fingerprint", all(all(r.get(k) == v for k, v in stamp.items()) for r in recs) and len(recs) == 3)
    ok, why = er.check_continuity(path, "t", "s1", [])
    check("a resume on the same system is allowed", ok)
    ok, why = er.check_continuity(path, "t", "s2", [])
    check("a resume after the system under test changed is REFUSED (the early and late cells would be different systems)", not ok and "different version" in why, why)
    legacy = os.path.join(tmp, "legacy.jsonl")
    with open(legacy, "w") as f:
        for r in recs:
            f.write(json.dumps({k: v for k, v in r.items() if k not in stamp}) + "\n")
    ok, why = er.check_continuity(legacy, "t", "s1", [])
    check("unstamped cells block a resume until someone attests to them", not ok and "attest" in why, why)
    ok, _ = er.check_continuity(legacy, "t", "s1", [{"attest": True, "pass": "t", "system_sha": "s1"}])
    check("... and an attestation for THIS system unblocks it", ok)
    ok, _ = er.check_continuity(legacy, "t", "s1", [{"attest": True, "pass": "t", "system_sha": "OTHER"}])
    check("... but an attestation for a different system does not", not ok)

with tempfile.TemporaryDirectory() as tmp:
    log = os.path.join(tmp, "RUNS.jsonl")
    open(log, "w").write(json.dumps({"ts": "2026-11-05T13:19:07", "commit": "c0c0", "pass": "full", "runs": 69, "stopped": "daily_quota"}) + "\n")
    sysf = lambda root=None: None
    out = []
    only_provider = lambda c0: ["llm_provider.py", "exp_run.py", "test_exp.py"]
    additions = lambda c0, name: ["+def last_rate_limit():", "+    return None"]
    rc = er.attest("full", changed=only_provider, diff=additions, log_path=log, out=out.append)
    recs_ = [json.loads(l) for l in open(log)]
    check("attest: harness-only changes plus pure additions to the provider are vouched for, with what was checked recorded",
          rc == 0 and recs_[-1]["attest"] and recs_[-1]["system_sha"] == er.fingerprint() and recs_[-1]["verified"]["llm_provider_pure_additions"] == 2, out)
    n_before = len(open(log).readlines())
    rc = er.attest("full", changed=lambda c0: ["clean_read.py"], diff=additions, log_path=log, out=out.append)
    check("attest REFUSES when a file the experiment imports (the reader) changed", rc == 2 and len(open(log).readlines()) == n_before and "clean_read.py" in out[-1], out[-1])
    rc = er.attest("full", changed=only_provider, diff=lambda c0, n: ["-    _MAX_RETRY_WAIT_S = 20.0", "+    _MAX_RETRY_WAIT_S = 30.0"], log_path=log, out=out.append)
    check("attest REFUSES when the provider diff removes or edits a line (a retry or model setting could have moved)", rc == 2 and "removed or edited" in out[-1], out[-1])
    check("attest refuses a pass that has no session in the log", er.attest("nope", changed=only_provider, diff=additions, log_path=log, out=out.append) == 2)

ok_cells = [dict(rec("a", "clean", 1, "correct"), system_sha="s1", model="m", temperature=0.1), dict(rec("a", "dirty", 1, "correct"), system_sha="s1", model="m", temperature=0.1)]
check("validity: one system, one model, one temperature -> valid", ea.validity(ok_cells, [])[0])
mixed = ok_cells + [dict(rec("b", "clean", 1, "correct"), system_sha="s2", model="m", temperature=0.1)]
check("validity: cells from two versions of the system -> INVALID", not ea.validity(mixed, [])[0])
check("validity: two models or two temperatures -> INVALID", not ea.validity(ok_cells + [dict(rec("b", "clean", 1, "correct"), system_sha="s1", model="other", temperature=0.1)], [])[0]
      and not ea.validity(ok_cells + [dict(rec("b", "clean", 1, "correct"), system_sha="s1", model="m", temperature=0.5)], [])[0])
legacy_cells = [rec("z", "clean", 1, "correct")]
check("validity: unstamped cells need an attestation for the system that produced the rest", not ea.validity(legacy_cells + ok_cells, [])[0]
      and ea.validity(legacy_cells + ok_cells, [{"system_sha": "s1"}])[0] and not ea.validity(legacy_cells + ok_cells, [{"system_sha": "s9"}])[0])
# ── dry run: tokens, not correctness ─────────────────────────────────────────
rows = [{"outcome": "wrong", "calls": 1, "tokens_prompt": 1500, "tokens_completion": 300}, {"outcome": "correct", "calls": 2, "tokens_prompt": 3000, "tokens_completion": 700},
        {"outcome": "infra_error", "calls": 0, "tokens_prompt": 0, "tokens_completion": 0}]
s = er.dry_summary(rows, 200000)
check("dry-run arithmetic: tokens per call, calls per run, and the daily budget they imply", s["usable"] and s["tokens_per_call"] == 1833 and s["calls_per_run"] == 1.5
      and s["recommended_daily_call_limit"] == int(0.9 * 200000 / (5500 / 3)) and s["runs_per_day"] == int(0.9 * 200000 / 2750), s)
check("a dry run with no usable runs says so instead of dividing by zero", er.dry_summary([{"outcome": "infra_error", "calls": 0, "tokens_prompt": 0, "tokens_completion": 0}], 200000) == {"usable": False})
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    fake = lambda data, q, customers, session: er.mechanical_solver(data, q, customers, session)
    er.SOLVERS["llm"] = fake
    before = set(os.listdir(er.RESULTS_DIR)) if os.path.isdir(er.RESULTS_DIR) else set()
    er.dry_run(questions, cases, 200000)
    after = set(os.listdir(er.RESULTS_DIR)) if os.path.isdir(er.RESULTS_DIR) else set()
text = buf.getvalue().lower()
check("the dry run never prints or stores correctness", "correct" not in text and "wrong" not in text and before == after, text[:200])

# ── collapsing repeats, headroom, decisions ──────────────────────────────────
def rec(qid, arm, rep, outcome, note=False, tier="single", kind="value", tag="x"):
    return {"qid": qid, "arm": arm, "repeat": rep, "outcome": outcome, "note": note, "tier": tier, "kind": kind, "tag": tag, "actual": "v", "solver": "llm",
            "pass": "full", "calls": 1, "tokens_prompt": 10, "tokens_completion": 5}
m, _ = ea.collapse([rec("a", "dirty", 1, "correct"), rec("a", "dirty", 2, "wrong"), rec("a", "dirty", 3, "correct"),
                    rec("b", "dirty", 1, "correct"), rec("b", "dirty", 2, "wrong"),
                    rec("c", "dirty", 1, "correct"), rec("c", "dirty", 2, "infra_error"), rec("c", "dirty", 3, "infra_error"),
                    rec("d", "dirty", 1, "crash"), rec("d", "dirty", 2, "crash"), rec("d", "dirty", 3, "correct")])
check("majority: strictly more than half of the non-infra repeats; a tie counts as wrong", m[("a", "dirty")]["correct"] and not m[("b", "dirty")]["correct"])
check("majority: infra repeats are dropped, not counted as wrong", m[("c", "dirty")]["n"] == 1 and m[("c", "dirty")]["correct"])
check("a crash is a wrong answer, not an infra error", m[("d", "dirty")]["crash"] and not m[("d", "dirty")]["correct"])

def synth(n_single, dirty_wrong_single, n_comp=10, dirty_wrong_comp=0, clean_wrong=0, stage0=None):
    qs, recs = [], []
    for i in range(n_single + n_comp):
        tier = "single" if i < n_single else "compound"
        q = {"id": f"q{i}", "tag": f"t{i % 5}", "tier": tier, "kind": "value", "expected": "1", "question": "?"}
        qs.append(q)
        wrong_d = (i < dirty_wrong_single) if tier == "single" else (i - n_single < dirty_wrong_comp)
        recs.append(rec(q["id"], "clean", 1, "wrong" if i < clean_wrong else "correct", tier=tier, tag=q["tag"]))
        recs.append(rec(q["id"], "dirty", 1, "wrong" if wrong_d else "correct", tier=tier, tag=q["tag"]))
        recs.append(rec(q["id"], "stage0", 1, "wrong" if wrong_d else "correct", tier=tier, tag=q["tag"]))
    return qs, recs

def verdict(**kw):
    qs, recs = synth(**kw)
    m, _ = ea.collapse(recs)
    return ea.decide(ea.headroom(qs, m, "single"), ea.headroom(qs, m, "compound")), ea.headroom(qs, m, "single")
v, h = verdict(n_single=30, dirty_wrong_single=15, dirty_wrong_comp=5)
check("a large gap whose interval excludes zero -> BUILD the validators", v.startswith("BUILD") and h["H"] == 50.0 and h["lo"] > 0, (v, h))
v, h = verdict(n_single=30, dirty_wrong_single=0, dirty_wrong_comp=0)
check("no gap anywhere (single AND compound) -> DEPRIORITIZE", v.startswith("DEPRIORITIZE"), v)
v, _ = verdict(n_single=30, dirty_wrong_single=0, dirty_wrong_comp=6)
check("no gap in the single tier but a gap in the compound tier -> INCONCLUSIVE, not deprioritize", v.startswith("INCONCLUSIVE"), v)
v, h = verdict(n_single=30, dirty_wrong_single=4)
check("13 points over 30 questions with an interval excluding zero meets the BUILD rule exactly as registered", v.startswith("BUILD") and 10 <= h["H"] and h["lo"] > 0, (v, h))
v, h = verdict(n_single=12, dirty_wrong_single=3)
check("a big point estimate on few questions, whose interval includes zero -> ADD repeats or questions", v.startswith("ADD") and h["H"] == 25.0 and h["lo"] <= 0, (v, h))
qs, recs = synth(n_single=20, dirty_wrong_single=10, clean_wrong=10)
m, _ = ea.collapse(recs)
h = ea.headroom(qs, m, "single")
check("headroom is conditioned on questions the CLEAN arm got right (model errors do not leak in)", h["n"] == 10 and h["H"] == 0.0, h)
check("the bootstrap is deterministic (fixed seed)", ea.bootstrap([0, 1, 1, 0, 1] * 6) == ea.bootstrap([0, 1, 1, 0, 1] * 6))

def wiring(stage0_better=0, stage0_worse=0, new_crash=False, struct_worse=False):
    qs, recs = [], []
    for i in range(12):
        structural = i < 4
        q = {"id": f"s{i}", "tag": ["footer", "total", "header", "ragged"][i] if structural else f"t{i}", "tier": "single", "kind": "structural" if structural else "value", "expected": "1", "question": "?"}
        qs.append(q)
        d, s = "correct", "correct"
        if structural and struct_worse and i == 0:
            s = "wrong"
        if not structural and i - 4 < stage0_worse:
            s = "wrong"
        if not structural and 4 <= i - 4 < 4 + stage0_better:
            d = "wrong"
        if structural and new_crash and i == 1:
            s = "crash"
        recs += [rec(q["id"], "clean", 1, "correct"), rec(q["id"], "dirty", 1, d), rec(q["id"], "stage0", 1, s)]
    m, _ = ea.collapse(recs)
    return ea.stage0_wiring(qs, m)
check("stage 0 wires in when it matches or beats dirty, with no new crashes", wiring()["verdict"].startswith("WIRE"))
check("one question net worse on non-structural questions is tolerated", wiring(stage0_worse=1)["verdict"].startswith("WIRE"))
check("two questions net worse is not", wiring(stage0_worse=2)["verdict"].startswith("DO NOT"))
check("worse questions are offset by better ones (it is NET)", wiring(stage0_worse=2, stage0_better=1)["verdict"].startswith("WIRE"))
check("a new crash blocks wiring", wiring(new_crash=True)["verdict"].startswith("DO NOT") and wiring(new_crash=True)["new_crashes"])
check("being worse than dirty on a structural defect type blocks wiring", wiring(struct_worse=True)["verdict"].startswith("DO NOT"))

# ── the report: shakedown banner, no verdict, the clean-wrong list ───────────
qs, recs = synth(n_single=30, dirty_wrong_single=15, clean_wrong=2)
for r in recs:
    r["pass"] = "shakedown1"
txt = ea.report(recs, qs, decide_flag=True)
check("a shakedown is labelled as one and evaluates no decision rule", "SHAKEDOWN" in txt and "not evaluated (shakedown)" in txt and "DECISION (pre-registered" not in txt)
check("the clean-arm failures are listed before any headroom is read", txt.index("3. questions the CLEAN arm") < txt.index("4. headroom") and "q0" in txt.split("4. headroom")[0])
full = [dict(r, pass_="full") for r in recs]
for r in full:
    r["pass"] = "full"
    r["repeat"] = 3
txt = ea.report(full, qs, decide_flag=True)
check("a full run (3 repeats, both tiers) evaluates the pre-registered rules", "DECISION (pre-registered rules, full run)" in txt and "validators:" in txt)
g = ea.gap_closed(qs, ea.collapse(recs)[0], "value")
check("gap closed is reported only where clean - dirty is at least 20 points", g["closed"] is not None and ea.gap_closed(synth(n_single=30, dirty_wrong_single=2)[0], ea.collapse(synth(n_single=30, dirty_wrong_single=2)[1])[0], "value")["closed"] is None)

qs_v, recs_v = synth(n_single=30, dirty_wrong_single=15)
for r in recs_v:
    r["pass"], r["repeat"] = "full", 3
txt = ea.report(recs_v, qs_v, decide_flag=True, check=(False, ["   INVALID: test"]))
check("a decision is REFUSED on an invalid run, however clear the gap looks", "DECISION: refused" in txt and "BUILD" not in txt)

print()
if FAIL:
    print(f"{len(FAIL)} check(s) failed: {FAIL}")
    sys.exit(1)
print("all checks passed")
