"""
Evals as a release gate: recorded model responses, trust scoring, pass/fail.

Run from the "AST 2.0" folder:
    python -m tests.evaluation.test_eval_gate

Uses a STUB model throughout (no network, no quota). Pins four properties:
  1. Recorded model responses replay exactly, with the model unreachable.
  2. A prompt that changed since recording is a loud "re-record" failure,
     never a silent pass and never a silent live call.
  3. Each answer is scored by TRUST: the number that matters is how many
     wrong answers were shown to the user as cross-checked.
  4. The gate fails on a regression and passes otherwise.
"""
from __future__ import annotations

import json
import os
import tempfile

import pandas as pd

FAIL: list[str] = []


def check(label: str, condition: bool, detail: str = ""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(label)


CSV = b"region,product,revenue\nnorth,a,100\nsouth,b,250\nnorth,a,50\neast,c,75\nsouth,b,25\n"
QUESTIONS = [
    # id, question, expected, type
    ("G1", "What is the revenue per region added up, for north only?", "150", "number"),
    ("G2", "What is the combined revenue of the south region?", "275", "number"),
    ("G3", "What is the revenue of the east region?", "75", "number"),
    ("G4", "Give me the largest single revenue value in the data", "250", "number"),
]


def plan_sum(region):
    return json.dumps({"select": [{"agg": "sum", "column": "revenue", "as": "total"}],
                       "filters": [{"column": "region", "op": "=", "value": region}]})


def stub_model(prompt, temperature=0.1, **kw):
    """Plays the live model. G1 right plan; G2 a plan for the WRONG question
    (both engines will agree on the wrong number); G3 right plan; G4 says
    unsupported, then writes Python."""
    stub_model.calls += 1
    is_plan = "JSON QUERY PLAN" in prompt
    if "north only" in prompt:
        return plan_sum("north")
    if "south region" in prompt:
        return plan_sum("north")          # wrong region: a verified WRONG answer
    if "east region" in prompt:
        return plan_sum("east")
    if "largest single" in prompt:
        return '{"unsupported": true}' if is_plan else "result = df['revenue'].max()"
    return "result = None"
stub_model.calls = 0


def unreachable_model(*a, **k):
    unreachable_model.calls += 1
    raise AssertionError("the live model was called during replay")
unreachable_model.calls = 0


def main():
    os.environ.pop("AST_QUERY_PLAN", None)
    from evaluation import eval_gate as gate
    from evaluation.llm_cassette import Cassette, normalize_prompt
    from evaluation.eval_harness import Question
    from security.prompt_injection_guard import wrap_untrusted

    questions = [Question(id=i, question=q, expected=e, answer_type=t) for i, q, e, t in QUESTIONS]
    tmp = tempfile.mkdtemp(prefix="ast_gate_")
    cpath = os.path.join(tmp, "cassette.json")

    print("=" * 70)
    print("PART 1 - prompts that differ only by the random fence tag are the same prompt")
    print("=" * 70)
    a, b = wrap_untrusted("col: x", source_label="t"), wrap_untrusted("col: x", source_label="t")
    check("two fenced prompts really do differ (random tag)", a != b)
    check("... and normalise to the same text", normalize_prompt(a) == normalize_prompt(b))
    check("a real content change is still a different prompt",
          normalize_prompt(a) != normalize_prompt(wrap_untrusted("col: y", source_label="t")))

    print("\n" + "=" * 70)
    print("PART 2 - record, then replay with the model unreachable")
    print("=" * 70)
    rec = Cassette(cpath, mode="record", live=stub_model)
    with rec.installed():
        recorded = gate.run_questions(CSV, "t.csv", questions)
    rec.save()
    check("recording called the live model", stub_model.calls >= 4, str(stub_model.calls))
    check("the cassette file was written", os.path.exists(cpath) and len(json.load(open(cpath))["entries"]) >= 4)

    rep = Cassette(cpath, mode="replay", live=unreachable_model)
    with rep.installed():
        replayed = gate.run_questions(CSV, "t.csv", questions)
    check("replay never touched the live model", unreachable_model.calls == 0)
    check("replay had no cassette misses", rep.misses == [], repr(rep.misses)[:120])
    check("replay gives the same outcome for every question",
          [(r.id, r.outcome, r.actual) for r in recorded] == [(r.id, r.outcome, r.actual) for r in replayed])
    rep2 = Cassette(cpath, mode="replay")
    with rep2.installed():
        again = gate.run_questions(CSV, "t.csv", questions)
    check("replay is deterministic run to run", [r.actual for r in again] == [r.actual for r in replayed])

    print("\n" + "=" * 70)
    print("PART 3 - every answer is scored by trust")
    print("=" * 70)
    by = {r.id: r for r in replayed}
    trust = gate.trust_summary(replayed)
    check("G1: right plan -> verified and correct", gate.trust_of(by["G1"]) == "verified_correct", gate.trust_of(by["G1"]))
    check("G2: plan for the wrong question -> VERIFIED BUT WRONG (the dangerous one)",
          gate.trust_of(by["G2"]) == "verified_wrong", gate.trust_of(by["G2"]))
    check("G4: fell back to generated code -> correct but unverified",
          gate.trust_of(by["G4"]) == "unverified_correct" and by["G4"].method == "generated_python", gate.trust_of(by["G4"]))
    check("the summary counts them", trust["verified_correct"] == 2 and trust["verified_wrong"] == 1
          and trust["unverified_correct"] == 1 and trust["correct"] == 3 and trust["asked"] == 4, repr(trust))
    check("the summary names the verified-wrong questions", trust["verified_wrong_ids"] == ["G2"])

    from analysis import query_plan as qp
    orig_exec = qp.execute_plan

    def buggy(plan_, df_):  # an engine bug: every filter dropped
        res = orig_exec(plan_, df_)
        nf = qp.Plan(select=plan_.select, filters=(), group_by=plan_.group_by, order_by=plan_.order_by, limit=plan_.limit)
        return qp.PlanResult(frame=orig_exec(nf, df_).frame, sql=res.sql, params=res.params, plan=plan_)
    qp.execute_plan = buggy
    try:
        with Cassette(cpath, mode="replay").installed():
            broken = gate.run_questions(CSV, "t.csv", questions)
    finally:
        qp.execute_plan = orig_exec
    bt = gate.trust_summary(broken)
    check("with a broken engine the plan answers are ABSTAINED, not shown wrong",
          bt["abstained"] == 3 and bt["verified_wrong"] == 0, repr(bt))
    check("abstained is its own outcome in the harness", {r.outcome for r in broken if r.id != "G4"} == {"abstained"})

    print("\n" + "=" * 70)
    print("PART 4 - a changed prompt is a loud re-record failure")
    print("=" * 70)
    changed = [Question(id="G1", question="What is the revenue per region added up, for north only??",
                        expected="150", answer_type="number")]
    miss = Cassette(cpath, mode="replay", live=unreachable_model)
    with miss.installed():
        res = gate.run_questions(CSV, "t.csv", changed)
    check("the miss is recorded", len(miss.misses) >= 1)
    check("the live model was still not called", unreachable_model.calls == 0)
    check("the question did not silently pass", not res[0].correct)

    print("\n" + "=" * 70)
    print("PART 5 - the gate decision")
    print("=" * 70)
    clean = dict(trust, verified_wrong=0, verified_wrong_ids=[], verified_correct=3)
    baseline = gate.baseline_from(clean)
    check("identical clean results pass", gate.gate_problems(clean, baseline, misses=0) == [])
    worse = dict(clean, correct=clean["correct"] - 1)
    check("fewer correct answers than the baseline fails", bool(gate.gate_problems(worse, baseline, misses=0)))
    check("any cassette miss fails", bool(gate.gate_problems(clean, baseline, misses=1)))
    better = dict(clean, correct=clean["correct"] + 1)
    check("an improvement passes", gate.gate_problems(better, baseline, misses=0) == [])
    check("a different number of questions fails (the suite changed; re-record)",
          bool(gate.gate_problems(dict(clean, asked=5), baseline, misses=0)))
    # The absolute rule: a wrong answer shown as cross-checked is never
    # acceptable, even when the baseline was recorded with the same defect.
    tolerant = gate.baseline_from(trust)                       # baseline says verified_wrong = 1
    problems = gate.gate_problems(trust, tolerant, misses=0)
    check("a verified-wrong answer fails even if the baseline already had it", bool(problems), repr(problems))
    check("... and the message names the question", any("G2" in p for p in problems), repr(problems))
    check("a result with zero verified-wrong passes against that same baseline",
          gate.gate_problems(dict(clean, correct=trust["correct"]), tolerant, misses=0) == [])

    print("\n" + "=" * 70)
    print("PART 6 - the command line: record a suite, replay it, exit codes")
    print("=" * 70)
    root = os.path.join(tmp, "proj")
    os.makedirs(os.path.join(root, "data"))
    open(os.path.join(root, "data", "t.csv"), "wb").write(CSV)
    pd.DataFrame([{"id": i, "question": q, "expected": e, "answer_type": t} for i, q, e, t in QUESTIONS]) \
        .to_csv(os.path.join(root, "data", "q.csv"), index=False)
    cfg = os.path.join(root, "gate.json")
    json.dump({"suites": [{"name": "tiny", "dataset": "data/t.csv", "questions": "data/q.csv"}]}, open(cfg, "w"))
    report_path = os.path.join(root, "last_report.md")

    def clean_model(prompt, temperature=0.1, **kw):   # like stub_model, but G2 gets the right plan
        return plan_sum("south") if "south region" in prompt else stub_model(prompt, temperature, **kw)

    code = gate.main(["--config", cfg])
    check("replay before any recording fails with a clear exit code", code == 2, str(code))

    code = gate.main(["--config", cfg, "--record"], live=stub_model)
    check("recording a suite with a verified-wrong answer exits 4", code == 4, str(code))
    check("... the cassette and baseline are still written, so it can be examined",
          os.path.exists(os.path.join(root, "cassettes", "tiny.json")) and os.path.exists(os.path.join(root, "baseline.json")))
    code = gate.main(["--config", cfg], live=unreachable_model)
    check("... and replaying it fails (exit 1) until the defect is fixed", code == 1, str(code))

    code = gate.main(["--config", cfg, "--record"], live=clean_model)
    check("recording a clean suite succeeds", code == 0, str(code))
    code = gate.main(["--config", cfg], live=unreachable_model)
    check("replay passes against its own baseline", code == 0, str(code))

    rows = {l.split("|")[1].strip(): l for l in open(report_path, encoding="utf-8").read().splitlines()
            if l.startswith("| G")}
    check("the report lists every question, not only failures", sorted(rows) == ["G1", "G2", "G3", "G4"], str(sorted(rows)))
    check("a correct plan answer shows its trust label and route",
          "verified_correct" in rows.get("G1", "") and "query_plan" in rows.get("G1", ""), rows.get("G1", ""))
    check("a correct generated-code answer is shown as unverified",
          "unverified_correct" in rows.get("G4", "") and "generated_python" in rows.get("G4", ""), rows.get("G4", ""))
    check("expected and actual values are shown", "275" in rows.get("G2", ""), rows.get("G2", ""))

    bl = json.load(open(os.path.join(root, "baseline.json")))
    bl["suites"]["tiny"]["correct"] += 1
    json.dump(bl, open(os.path.join(root, "baseline.json"), "w"))
    code = gate.main(["--config", cfg], live=unreachable_model)
    check("replay fails (exit 1) when results are below the baseline", code == 1, str(code))

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE - {len(FAIL)} FAILURE(S): {FAIL}")
        raise SystemExit(1)
    print("DONE - all checks passed.")


if __name__ == "__main__":
    main()
