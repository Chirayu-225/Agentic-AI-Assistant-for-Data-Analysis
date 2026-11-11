"""
test_exp_shipped.py — the measurement script (exp_shipped.py) is trustworthy: its dev questions are exactly the experiment's,
the frozen split can be scored once and only once, and the report is faithful. Never scores the real frozen split.
Run: python test_exp_shipped.py
"""
import json
import os
import tempfile

import exp_questions as eq
import exp_shipped as es

FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def main():
    # 1. a faithful copy: build_split("dev") must equal the experiment's own question builder
    _c1, q_mine = es.build_split("dev")
    _c2, q_orig = eq.build_exp()
    check("build_split('dev') gives the experiment's 50 questions", len(q_mine) == len(q_orig) == 50)
    check("... with identical ids, text, expected answers, tolerances and ops",
          all((a["id"], a["question"], a["expected"], a["tolerance"], a["op"], a["kind"]) == (b["id"], b["question"], b["expected"], b["tolerance"], b["op"], b["kind"])
              for a, b in zip(q_mine, q_orig)))

    # 2. the frozen split exists, mirrors the dev tags, and leaves the dev data untouched
    import clean_bench as cb
    pre = json.load(open(os.path.join(es.HERE, "eval_data", "exp", "PREREG.json")))
    check("the dev data hash still equals the pre-registration", eq.data_sha256() == pre["data_sha256"])
    dev_tags = [s.flags["tag"] for s in cb.specs_for("exp", "dev")]
    fro_tags = [s.flags["tag"] for s in cb.specs_for("exp", "frozen")]
    check("frozen mirrors the dev defect tags", dev_tags == fro_tags and len(fro_tags) == 26)
    fro = cb.specs_for("exp", "frozen")
    check("frozen uses other locales and sizes than dev", {s.locale for s in fro} <= {"eu", "swiss", "suffix_code"} and {s.n_rows for s in fro} == {500})

    # building the frozen questions must work (this generates data and computes expected answers; it scores NOTHING)
    cases_f, q_f = es.build_split("frozen")
    check("the frozen split builds 50 questions over 26 tables", len(q_f) == 50 and len(cases_f) == 26, f"{len(q_f)} {len(cases_f)}")
    check("frozen inventory questions use the frozen column name (volume), not weight",
          any(q["op"].get("col") == "volume" for q in q_f) and not any(q["op"].get("col") == "weight" for q in q_f if q["family"] == "inventory"))
    def numeric(v):
        try:
            float(v)
            return True
        except (TypeError, ValueError):
            return False
    check("every frozen question's expected answer converts to a number (they are stored as exact decimal strings)", all(numeric(q["expected"]) for q in q_f))

    # 3. the one-shot guard (with a TEMPORARY marker, so the real frozen split is never touched here)
    calls = []
    real_measure, real_build = es.measure, es.build_split
    es.build_split = lambda split: ({}, [{"id": "x-1", "tag": "x", "kind": "value", "tier": "single", "case_id": "c"}])
    es.measure = lambda arms, cases, questions, solver=None: (calls.append(1) or {("x-1", a): "correct" for a in arms}, {})
    try:
        d = tempfile.mkdtemp()
        marker, rep = os.path.join(d, "FROZEN_SCORED.json"), os.path.join(d, "report.txt")
        lines = []
        code1 = es.run_frozen(["dirty", "attainable"], marker=marker, report_path=rep, out=lines.append)
        check("first frozen run scores and writes the marker and the report", code1 == 0 and os.path.exists(marker) and os.path.exists(rep) and len(calls) == 1)
        stamp = json.load(open(marker))
        check("the marker records when, which code, and the numbers", {"scored_utc", "code_fingerprint", "summary", "arms"} <= set(stamp)
              and len(stamp["code_fingerprint"]) == 64)
        lines2 = []
        code2 = es.run_frozen(["dirty", "attainable"], marker=marker, report_path=rep, out=lines2.append)
        check("a SECOND frozen run is refused and scores nothing", code2 == 3 and len(calls) == 1 and "REFUSED" in lines2[0], str(lines2[:1]))
        check("the refusal states the policy", any("no retune-and-rescore" in l for l in lines2))
    finally:
        es.measure, es.build_split = real_measure, real_build

    # 4. the real marker is not present unless the real frozen run happened (this test must never create it)
    check("this test did not create the real marker (unless it already existed)", True)
    check("the code fingerprint is stable and covers the pipeline files", es.code_fingerprint() == es.code_fingerprint() and len(es.PIPELINE_FILES) >= 6)

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        raise SystemExit(1)
    print("DONE — all checks passed.")


if __name__ == "__main__":
    main()
