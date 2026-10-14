"""
test_eval_harness.py — the evaluation harness itself: answer checking for
every answer type, tolerance edge cases, failure classification, Wilson
intervals, repeat-consistency, template generation, question-file
validation, and an end-to-end offline run with the shortcut solver.

Run: python3 test_eval_harness.py
"""

import json
import os
import shutil
import sys
import tempfile

os.environ.setdefault("AUDIT_LOG_DISABLED", "1")

import eval_harness as eh
from eval_harness import Question

FAIL = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, condition, detail=""):
    print(f"[{'OK' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(label)


def scalar(v):
    return {"success": True, "resultType": "scalar", "resultData": v}


def table(rows, total=None):
    r = {"success": True, "resultType": "table", "resultData": rows}
    if total is not None:
        r["totalRowCount"] = total
    return r


def main():
    # Guarded: on Windows the sandbox uses "spawn", which re-imports this file in
    # each worker; without the guard the whole test would re-run inside the worker.
    TMP = tempfile.mkdtemp(prefix="eval_test_")
    try:
        print("=" * 70)
        print("PART 1 — number answers and tolerance")
        print("=" * 70)
        q = Question("n", "?", "100", "number", 0.01)
        check("exact match is correct", eh.check_answer(q, scalar(100)).correct)
        check("within 1% is correct", eh.check_answer(q, scalar(100.9)).correct)
        check("outside 1% is wrong", not eh.check_answer(q, scalar(102)).correct)
        check("string number '1,000' is parsed", eh.check_answer(Question("n", "?", "1000", "number", 0), scalar("1,000")).correct)
        v = eh.check_answer(q, scalar(102))
        check("a wrong number is comparable (wrong_answer, not no_answer)", v.comparable and not v.correct)
        z = Question("z", "?", "0", "number", 0)
        check("expected=0, actual=0 is correct", eh.check_answer(z, scalar(0)).correct)
        check("expected=0, actual=5 is wrong (no relative-tolerance loophole)", not eh.check_answer(z, scalar(5)).correct)
        check("one-row, one-number table is accepted", eh.check_answer(q, table([{"total": 100}])).correct)
        v = eh.check_answer(q, table([{"a": 1}, {"a": 2}]))
        check("multi-row table for a number question is NOT guessed at", (not v.correct) and (not v.comparable))
        v = eh.check_answer(q, table([{"a": 1, "b": 2}]))
        check("one row with two numeric cells is ambiguous, not guessed", (not v.correct) and (not v.comparable))
        v = eh.check_answer(q, scalar("many"))
        check("non-numeric scalar is no_answer", (not v.correct) and (not v.comparable))
        check("a bool is not treated as a number", not eh.check_answer(Question("b", "?", "1", "number", 0), scalar(True)).correct)
        v = eh.check_answer(Question("bad", "?", "abc", "number"), scalar(1))
        check("a non-numeric expected value is reported, not crashed on", not v.correct and "not a number" in v.detail)

        print("\n" + "=" * 70)
        print("PART 2 — text / contains / rows / chart")
        print("=" * 70)
        t = Question("t", "?", "South", "text")
        check("text: case and whitespace insensitive", eh.check_answer(t, table([{"region": "  south ", "rev": 5}])).correct)
        check("text: wrong label fails", not eh.check_answer(t, table([{"region": "North", "rev": 5}])).correct)
        check("text: scalar label works", eh.check_answer(t, scalar("South")).correct)
        c = Question("c", "?", "South|East|North", "contains")
        rows = [{"region": r, "rev": i} for i, r in enumerate(["South", "East", "North"])]
        check("contains: all items present", eh.check_answer(c, table(rows)).correct)
        v = eh.check_answer(c, table(rows[:2]))
        check("contains: reports what is missing", (not v.correct) and "north" in v.detail.lower(), v.detail)
        r = Question("r", "?", "3", "rows")
        check("rows: counts a table", eh.check_answer(r, table(rows)).correct)
        check("rows: prefers totalRowCount when the table was truncated", eh.check_answer(Question("r", "?", "500", "rows"), table(rows, total=500)).correct)
        check("rows: scalar is not a table", not eh.check_answer(r, scalar(3)).comparable)
        ch = Question("ch", "?", "", "chart")
        check("chart: passes when a chart was produced", eh.check_answer(ch, {"success": True, "resultType": "chart"}).correct)
        check("chart: fails on a table", not eh.check_answer(ch, table(rows)).correct)

        print("\n" + "=" * 70)
        print("PART 3 — failure classification")
        print("=" * 70)
        cases = {
            "Security block: import is not allowed": "security_block",
            "No shortcut matched this question": "needs_llm",
            "Query exceeded the 2s execution limit and was stopped": "timeout",
            "Memory limit exceeded": "memory_limit",
            "Analysis pipeline failed: LLM unreachable": "pipeline_error",
            "KeyError: 'nope'": "execution_error",
        }
        for msg, want in cases.items():
            check(f"classify {msg[:42]!r} -> {want}", eh.classify_failure(msg) == want, eh.classify_failure(msg))

        print("\n" + "=" * 70)
        print("PART 4 — run_eval with fake solvers")
        print("=" * 70)
        qs = [Question("A", "total?", "10", "number", 0), Question("B", "avg?", "5", "number", 0),
              Question("C", "boom?", "1", "number", 0), Question("D", "fail?", "1", "number", 0),
              Question("E", "chart", "", "chart")]

        def fake(file_bytes, filename, question, session_id):
            if question == "total?":
                return scalar(10)
            if question == "avg?":
                return scalar(6)
            if question == "boom?":
                raise RuntimeError("solver crashed")
            if question == "fail?":
                return {"success": False, "error": "Security block: nope"}
            return {"success": True, "resultType": "chart"}

        seen = []
        res = eh.run_eval(b"", "x.csv", qs, fake, on_result=seen.append)
        by = {r.id: r for r in res}
        check("every question produced a result", len(res) == 5 and len(seen) == 5)
        check("A correct", by["A"].outcome == "correct" and by["A"].correct)
        check("B wrong_answer (executed but wrong)", by["B"].outcome == "wrong_answer" and by["B"].executed)
        check("C: a solver exception is a result, not a harness crash", by["C"].outcome == "solver_exception" and not by["C"].executed)
        check("D classified as security_block and not executed", by["D"].outcome == "security_block" and not by["D"].executed)
        check("E chart correct", by["E"].correct)

        s = eh.summarize(res)
        check("accuracy = 2/5", s["accuracy"] == 0.4, str(s["accuracy"]))
        check("execution success = 3/5", s["executionSuccess"] == 0.6)
        lo, hi = s["accuracy95"]
        check("the interval brackets the point estimate", lo < 0.4 < hi)
        check("outcome counts add up to the run count", sum(s["outcomes"].values()) == 5)

        print("\n" + "=" * 70)
        print("PART 5 — Wilson interval sanity")
        print("=" * 70)
        lo, hi = eh.wilson_interval(0, 10)
        check("0/10: lower bound 0, upper bound well above 0", lo == 0 and 0.2 < hi < 0.4, f"{lo},{hi}")
        lo, hi = eh.wilson_interval(10, 10)
        check("10/10: upper bound 1, lower bound well below 1", hi == 1 and 0.6 < lo < 0.8, f"{lo},{hi}")
        lo, hi = eh.wilson_interval(17, 20)
        check("17/20 matches the textbook Wilson interval (~0.64-0.95)", abs(lo - 0.640) < 0.01 and abs(hi - 0.948) < 0.01, f"{lo:.3f},{hi:.3f}")
        a = eh.wilson_interval(8, 10); b = eh.wilson_interval(80, 100)
        check("more data gives a narrower interval", (b[1] - b[0]) < (a[1] - a[0]))
        check("n=0 does not divide by zero", eh.wilson_interval(0, 0) == (0.0, 0.0))

        print("\n" + "=" * 70)
        print("PART 6 — --repeat consistency and effort")
        print("=" * 70)
        flip = {"n": 0}

        def flaky(file_bytes, filename, question, session_id):
            flip["n"] += 1
            return scalar(10 if flip["n"] % 2 else 11)   # alternates every call

        steady_q = Question("S", "steady?", "10", "number", 0, manual_seconds=60)
        flaky_q = Question("F", "flaky?", "10", "number", 0, manual_seconds=30)

        def mixed(fb, fn, question, sid):
            return scalar(10) if question == "steady?" else flaky(fb, fn, question, sid)

        res = eh.run_eval(b"", "x.csv", [steady_q, flaky_q], mixed, repeat=4)
        s = eh.summarize(res, repeat=4)
        check("8 runs recorded for 2 questions x 4", s["runs"] == 8 and s["questions"] == 2)
        seen_ids = []
        eh.run_eval(b"", "x.csv", [steady_q], lambda fb, fn, qq, sid: seen_ids.append(sid) or scalar(10), repeat=3, session_id="S")
        check("every run gets its own session id (so LLM code caches can't fake consistency)",
              len(set(seen_ids)) == 3, str(seen_ids))
        check("consistency = mean of per-question agreement (1.0 and 0.5 -> 0.75)", s["consistency"] == 0.75, str(s["consistency"]))
        check("effort compares manual vs assistant time", s["effort"] and s["effort"]["manualSecondsTotal"] == 90.0
              and s["effort"]["questionsWithManualTime"] == 2)
        relabel = iter([{"region": "South", "units": 39}, {"region": "South", "total_units": 39.0}, {"Region": " south ", "n": "39"}])
        res_rel = eh.run_eval(b"", "x.csv", [Question("T", "?", "South", "text")],
                              lambda fb, fn, qq, sid: table([next(relabel)]), repeat=3)
        check("same values under different column names / formatting count as CONSISTENT",
              eh.summarize(res_rel, repeat=3)["consistency"] == 1.0, str(eh.summarize(res_rel, repeat=3)["consistency"]))
        diff = iter([39, 40, 39])
        res_diff = eh.run_eval(b"", "x.csv", [Question("D", "?", "39", "number", 0)],
                               lambda fb, fn, qq, sid: scalar(next(diff)), repeat=3)
        check("genuinely different values are still inconsistent (2 of 3 agree -> 0.667)",
              abs(eh.summarize(res_diff, repeat=3)["consistency"] - 0.667) < 0.001)
        check("consistency is not reported for a single run", eh.summarize(res[:2], repeat=1)["consistency"] is None)

        print("\n" + "=" * 70)
        print("PART 7 — question files and template")
        print("=" * 70)
        qpath = os.path.join(TMP, "q.csv")

        def write(text):
            with open(qpath, "w", encoding="utf-8") as f:
                f.write(text)

        write("id,question,expected,answer_type\nQ1,How many?,24,number\n,Plot it,,chart\n")
        loaded = eh.load_questions(qpath)
        check("valid file loads (chart needs no expected)", len(loaded) == 2 and loaded[1].answer_type == "chart")
        check("missing id gets a default label", loaded[1].id == "Q2")
        check("default tolerance applied", loaded[0].tolerance == eh.DEFAULT_TOLERANCE)

        for label, body in (("placeholder expected", "question,expected\nHow many?,<your answer>\n"),
                            ("empty expected", "question,expected\nHow many?,\n"),
                            ("bad answer_type", "question,expected,answer_type\nHow many?,1,essay\n"),
                            ("no questions", "question,expected\n")):
            write(body)
            try:
                eh.load_questions(qpath); ok = False
            except ValueError:
                ok = True
            check(f"rejects {label}", ok)

        jpath = os.path.join(TMP, "q.json")
        with open(jpath, "w") as f:
            json.dump([{"question": "How many?", "expected": "5", "answer_type": "number"}], f)
        check("JSON question files load too", len(eh.load_questions(jpath)) == 1)

        data = os.path.join(HERE, "eval_data", "sample_sales.csv")
        cwd = os.getcwd(); os.chdir(TMP)
        try:
            tpl = eh.write_template(data, "questions_template.csv")
            text = open(tpl, encoding="utf-8").read()
        finally:
            os.chdir(cwd)
        check("template is written with example questions", text.count("\n") >= 7)
        check("template uses the dataset's real columns, not an ID column", "order_id" not in text.lower().split("\n")[2])
        try:
            eh.load_questions(os.path.join(TMP, "questions_template.csv")); ok = False
        except ValueError:
            ok = True
        check("an unedited template is refused (forces the user to supply ground truth)", ok)

        print("\n" + "=" * 70)
        print("PART 8 — end to end, offline, on the bundled sample data")
        print("=" * 70)
        qfile = os.path.join(HERE, "eval_data", "sample_questions.csv")
        out = os.path.join(TMP, "results")
        rc = eh.main(["--dataset", data, "--questions", qfile, "--solver", "shortcut", "--out", out])
        check("exit code is non-zero when accuracy < 100%", rc == 1)
        runs = sorted(os.listdir(out))
        check("one timestamped results folder is written", len(runs) == 1)
        folder = os.path.join(out, runs[0])
        files = set(os.listdir(folder))
        check("report.md, results.csv and summary.json are written", {"report.md", "results.csv", "summary.json"} <= files, str(files))
        summ = json.load(open(os.path.join(folder, "summary.json")))
        check("9 questions evaluated", summ["summary"]["questions"] == 9 if "summary" in summ else summ["questions"] == 9, str(list(summ)[:5]))
        report = open(os.path.join(folder, "report.md"), encoding="utf-8").read()
        check("report shows the confidence interval", "95%" in report)
        check("llm solver refuses to run without GROQ_API_KEY",
              (os.environ.pop("GROQ_API_KEY", None), eh.main(["--dataset", data, "--questions", qfile]))[1] == 2)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}")
        sys.exit(1)
    print("DONE — all checks passed.")



if __name__ == "__main__":
    main()
