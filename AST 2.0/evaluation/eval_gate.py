"""
The evaluation gate: a pass/fail decision on answer quality, from recorded
model responses.

    python -m evaluation.eval_gate              replay (free, offline, deterministic)
    python -m evaluation.eval_gate --record     re-record with the REAL model (uses quota)

WHAT IT DECIDES
    Each question has a ground-truth answer worked out by hand
    (evaluation/eval_harness.py). Every answer is scored twice: is it CORRECT,
    and what did the system CLAIM about it?

        verified_correct     shown as cross-checked, and right
        verified_wrong       shown as cross-checked, and WRONG   <- hazard H1
        unverified_correct   right, from generated code or a shortcut
        unverified_wrong     wrong, but never claimed to be checked
        abstained            withheld: "I can't verify this"
        failed               no answer (error, timeout, block)

    The gate fails (exit 1) when ANY answer is verified-but-wrong (an absolute
    rule: a baseline cannot make that acceptable), or when, compared with the
    recorded baseline, fewer answers are correct, the question set changed, or
    any prompt changed since recording (a cassette miss).

    Because replay is deterministic, a failure is caused by a code change -
    which is exactly when a release should stop.

EXIT CODES   0 pass   1 gate failed   2 not recorded yet / bad config   3 recording incomplete
             4 recorded, but the recording contains verified-but-wrong answers
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import eval_harness as eh
from evaluation.llm_cassette import Cassette

TRUST_KEYS = ("verified_correct", "verified_wrong", "unverified_correct", "unverified_wrong", "abstained", "failed")
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "eval_data", "gate", "gate.json")


def run_questions(file_bytes: bytes, filename: str, questions: list) -> list:
    """The real /analyze pipeline for each question (model calls go wherever
    the installed cassette sends them)."""
    return eh.run_eval(file_bytes, filename, questions, solver=eh.llm_solver, session_id="gate")


def trust_of(r) -> str:
    if r.outcome == "abstained":
        return "abstained"
    if not r.executed:
        return "failed"
    if r.verification == "verified":
        return "verified_correct" if r.correct else "verified_wrong"
    return "unverified_correct" if r.correct else "unverified_wrong"


def trust_summary(results: list) -> dict:
    scored = [r for r in results if r.outcome != "infra_error"]
    out = {k: 0 for k in TRUST_KEYS}
    for r in scored:
        out[trust_of(r)] += 1
    out["asked"] = len(scored)
    out["correct"] = sum(1 for r in scored if r.correct)
    out["infra_errors"] = len(results) - len(scored)
    out["verified_wrong_ids"] = [r.id for r in scored if trust_of(r) == "verified_wrong"]
    out["by_method"] = {}
    for r in scored:
        m = out["by_method"].setdefault(r.method or "(none)", {"asked": 0, "correct": 0})
        m["asked"] += 1
        m["correct"] += int(r.correct)
    return out


def baseline_from(trust: dict) -> dict:
    return {"asked": trust["asked"], "correct": trust["correct"], "verified_wrong": trust["verified_wrong"]}


def gate_problems(trust: dict, baseline: dict, misses: int) -> list:
    problems = []
    if misses:
        problems.append(f"{misses} prompt(s) changed since recording; re-record with --record (uses the real model)")
    if trust["asked"] != baseline["asked"]:
        problems.append(f"the suite now has {trust['asked']} question(s), the baseline has {baseline['asked']}; re-record")
    if trust["correct"] < baseline["correct"]:
        problems.append(f"correct answers dropped from {baseline['correct']} to {trust['correct']}")
    # Absolute, not relative: hazard H1 is never "no worse than last time".
    if trust["verified_wrong"]:
        problems.append(f"{trust['verified_wrong']} answer(s) shown as cross-checked are WRONG: "
                        f"{trust.get('verified_wrong_ids', [])}")
    return problems


def _report(rows: list) -> str:
    L = ["# Evaluation gate", "",
         "| Suite | Asked | Correct | Verified + correct | Verified + WRONG | Unverified correct | Unverified wrong | Abstained | Failed | Gate |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, t, problems, _ in rows:
        L.append(f"| {name} | {t['asked']} | {t['correct']} | {t['verified_correct']} | {t['verified_wrong']} | "
                 f"{t['unverified_correct']} | {t['unverified_wrong']} | {t['abstained']} | {t['failed']} | "
                 f"{'FAIL' if problems else 'pass'} |")
    for name, t, problems, results in rows:
        L += ["", f"## {name}", ""]
        L += [f"- GATE: {p}" for p in problems] or ["- Gate passed."]
        L.append("- By route: " + ", ".join(f"{m} {v['correct']}/{v['asked']}" for m, v in sorted(t["by_method"].items())))
        if results:
            L += ["", "| ID | Result | Trust | Route | Question | Expected | Got / reason |", "|---|---|---|---|---|---|---|"]
            for r in results:
                got = ((r.actual if r.correct else (r.actual or r.detail)) or "").replace("|", "/")[:90]
                L.append(f"| {r.id} | {'ok' if r.correct else 'WRONG'} | {trust_of(r)} | {r.method or '-'} | "
                         f"{r.question[:70].replace('|', '/')} | {str(r.expected).replace('|', '/')[:40]} | {got} |")
    L += ["", "Verified + WRONG is the number that matters: answers the system told the user it had cross-checked, "
              "which were wrong. The cross-check proves the plan was computed correctly, not that the plan "
              "matches the question.", ""]
    return "\n".join(L)


def main(argv: Optional[list] = None, live: Optional[Callable] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluation gate (replay by default).")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--record", action="store_true", help="call the REAL model and re-record cassettes + baseline")
    ap.add_argument("--suite", help="only this suite")
    args = ap.parse_args(argv)

    if not os.path.exists(args.config):
        print(f"[gate] no config at {args.config}")
        return 2
    root = os.path.dirname(os.path.abspath(args.config))
    with open(args.config, encoding="utf-8") as f:
        suites = json.load(f)["suites"]
    if args.suite:
        suites = [s for s in suites if s["name"] == args.suite]
    baseline_path = os.path.join(root, "baseline.json")
    baseline = {"suites": {}}
    if os.path.exists(baseline_path):
        with open(baseline_path, encoding="utf-8") as f:
            baseline = json.load(f)

    rows, exit_code = [], 0
    for s in suites:
        name = s["name"]
        dataset = os.path.normpath(os.path.join(root, s["dataset"]))
        questions = eh.load_questions(os.path.normpath(os.path.join(root, s["questions"])))
        cpath = os.path.join(root, "cassettes", f"{name}.json")
        with open(dataset, "rb") as f:
            data = f.read()

        if args.record:
            cassette = Cassette(cpath, mode="record", live=live)
        else:
            if not os.path.exists(cpath) or name not in baseline["suites"]:
                print(f"[gate] suite {name!r} has not been recorded yet; run with --record")
                return 2
            cassette = Cassette(cpath, mode="replay")
        print(f"[gate] {name}: {len(questions)} question(s), {'RECORDING (real model)' if args.record else 'replay'}", flush=True)
        with cassette.installed():
            results = run_questions(data, os.path.basename(dataset), questions)
        trust = trust_summary(results)

        if args.record:
            if trust["infra_errors"]:
                print(f"[gate] {name}: {trust['infra_errors']} question(s) hit a provider limit; NOT saved. "
                      f"Re-run --record --suite {name} when the quota resets.")
                exit_code = 3
                rows.append((name, trust, ["recording incomplete (provider limit)"], results))
                continue
            cassette.save()
            baseline["suites"][name] = baseline_from(trust)
            problems = []
            if trust["verified_wrong"]:
                print(f"[gate] WARNING {name}: {trust['verified_wrong']} answer(s) are shown as cross-checked but are "
                      f"WRONG: {trust['verified_wrong_ids']}. Recorded so they can be examined; replay will FAIL "
                      f"until they are fixed.")
                problems = [f"{trust['verified_wrong']} answer(s) shown as cross-checked are WRONG: "
                            f"{trust['verified_wrong_ids']}"]
                exit_code = max(exit_code, 4)
        else:
            problems = gate_problems(trust, baseline["suites"][name], len(cassette.misses))
            if problems:
                exit_code = max(exit_code, 1)
        rows.append((name, trust, problems, results))
        print(f"[gate] {name}: correct {trust['correct']}/{trust['asked']}, verified+wrong {trust['verified_wrong']}, "
              f"abstained {trust['abstained']}, failed {trust['failed']} -> {'FAIL' if problems else 'ok'}", flush=True)
        for p in problems:
            print(f"[gate]   {p}")

    if args.record:  # only suites that recorded completely were updated above
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=1, sort_keys=True)
    with open(os.path.join(root, "last_report.md"), "w", encoding="utf-8") as f:
        f.write(_report(rows))
    print(f"[gate] {'PASS' if exit_code == 0 else 'FAIL'} (exit {exit_code}); report: {os.path.join(root, 'last_report.md')}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
