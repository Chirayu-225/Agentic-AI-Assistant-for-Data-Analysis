"""
exp_shipped.py — how much of the dirty-data gap does the SHIPPED pipeline close? No Groq, no network.

The experiment (exp_run.py) asked a model the same question on three versions of each table. The model adds noise, and
Groq calls cost time, so this script isolates the TABLE: it scores each arm's table with the experiment's own mechanical
solver (exp_run.mechanical_solver: the reference computation, run on whatever table the arm hands over) and the
experiment's own checker (exp_run.classify). The answer is right exactly when the table is right.

  dirty       the file as it arrived (the old reader + the rule cleaner, inside the solver)
  stage0      the experiment's Stage 0 arm (reader + cleaner + CSV, then the cleaner again)
  bridge      the shipped structure-only bridge (stage0_bridge.py), when present
  shipped     bridge + rule cleaner + value validators (no fill)
  shipped_fill  the same with the agent's fill step (fill_missing.py) before the validators
  attainable  the ceiling: every recoverable defect repaired, every unrecoverable cell blanked

THIS IS NOT A PRE-REGISTERED RUN. It uses the experiment's dev tables, which the validators were written while looking at.
Treat the dev numbers as a development signal, never as a held-out result.  Usage:
    python exp_shipped.py                      # every arm that exists
    python exp_shipped.py --arms dirty,shipped
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import sys

import clean_bench as cb
import exp_questions as eq
import exp_run as er

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_MARKER = os.path.join(HERE, "eval_data", "exp", "FROZEN_SCORED.json")
FROZEN_REPORT = os.path.join(HERE, "eval_results", "exp", "frozen-shipped.report.txt")
# The files whose behaviour the shipped arms depend on: the fingerprint written beside the frozen score proves which code was scored.
PIPELINE_FILES = ("value_validators.py", "stage0_bridge.py", "shipped_pipeline.py", "cleaning.py", "file_reader.py",
                  "missing_tokens.py", "clean_read.py", "fill_missing.py")

KNOWN_ARMS = ("dirty", "stage0", "bridge", "shipped", "shipped_fill", "attainable")


def table_for(arm: str, files: dict, doc: dict):
    """-> (csv bytes or None, why-not). Arms whose module is not built yet raise LookupError, so the caller can skip them."""
    if arm == "dirty":
        return files["data.csv"], ""
    if arm == "attainable":
        return eq.attainable_csv(files, doc), ""
    if arm == "stage0":
        from clean_adapter_stage0 import cleaned_csv_bytes
        return cleaned_csv_bytes(files["data.csv"])
    if arm == "bridge":
        try:
            import stage0_bridge
        except ImportError:
            raise LookupError("stage0_bridge.py is not built yet") from None
        return stage0_bridge.bridge_bytes(files["data.csv"]), ""
    if arm == "shipped":
        try:
            import shipped_pipeline
        except ImportError:
            raise LookupError("shipped_pipeline.py is not built yet") from None
        return shipped_pipeline.shipped_csv_bytes(files["data.csv"]), ""
    if arm == "shipped_fill":
        try:
            import shipped_pipeline
        except ImportError:
            raise LookupError("shipped_pipeline.py is not built yet") from None
        return shipped_pipeline.shipped_csv_bytes(files["data.csv"], fill=True), ""
    raise ValueError(arm)


def measure(arms, cases=None, questions=None, solver=None):
    """-> ({(question id, arm): outcome}, {arm: why it was skipped})."""
    if cases is None or questions is None:
        cases, questions = eq.build_exp()
    solver = solver or er.mechanical_solver
    out, skipped = {}, {}
    for arm in arms:
        for q in questions:
            _spec, files, doc = cases[q["case_id"]]
            try:
                data, err = table_for(arm, files, doc)
            except LookupError as e:
                skipped[arm] = str(e)
                break
            except Exception as e:  # noqa: BLE001  an arm that crashes is a crash, not a skipped arm
                out[(q["id"], arm)] = "crash"
                continue
            if data is None:
                out[(q["id"], arm)] = "crash"
                continue
            customers = files.get("customers.csv")
            result = solver(data, q, customers, f"shipped-{q['id']}-{arm}")
            outcome, _actual = er.classify(q, result)
            out[(q["id"], arm)] = outcome
    return out, skipped


def report(out: dict, skipped: dict, arms, questions) -> str:
    arms = [a for a in arms if a not in skipped]
    by_tag = collections.OrderedDict()
    for q in questions:
        by_tag.setdefault(q["tag"], []).append(q)
    lines = ["per defect type (correct / questions; the table is scored, not a model)", ""]
    lines.append(f"  {'defect':<11}{'kind':<11}{'tier':<9}{'n':>2}  " + "".join(f"{a:>12}" for a in arms))
    for tag, qs in by_tag.items():
        cells = []
        for a in arms:
            ok = sum(out.get((q["id"], a)) == "correct" for q in qs)
            cells.append(f"{ok}/{len(qs)}")
        lines.append(f"  {tag:<11}{qs[0]['kind']:<11}{qs[0]['tier']:<9}{len(qs):>2}  " + "".join(f"{c:>12}" for c in cells))
    lines += ["", "overall"]
    for kind in ("structural", "value"):
        qs = [q for q in questions if q["kind"] == kind]
        if not qs:
            continue
        lines.append(f"  {kind:<11}(n={len(qs)})  " + "   ".join(
            f"{a} {100 * sum(out.get((q['id'], a)) == 'correct' for q in qs) / len(qs):.0f}%" for a in arms))
    allq = questions
    if not allq:
        return "\n".join(lines)
    lines.append(f"  {'all':<11}(n={len(allq)})  " + "   ".join(
        f"{a} {100 * sum(out.get((q['id'], a)) == 'correct' for q in allq) / len(allq):.0f}%" for a in arms))
    if skipped:
        lines += ["", "skipped: " + "; ".join(f"{a} ({why})" for a, why in skipped.items())]
    return "\n".join(lines)


def build_split(split: str):
    """-> (cases, questions) for the experiment's question set on `split`. For "dev" this is exactly eq.build_exp()
    (test_exp_shipped.py pins that); "frozen" is the held-out dialect split added to clean_bench.PROFILES["exp"]."""
    cases, questions = {}, []
    for spec in cb.specs_for("exp", split):
        files, doc = cb.build_case(spec)
        cases[spec.case_id] = (spec, files, doc)
        tag = spec.flags["tag"]
        tier, kind, ops = eq.PLAN[tag]
        import csv
        import io
        full = list(csv.DictReader(io.StringIO(files["truth.csv"].decode("utf-8"))))
        rows = eq.attainable_rows(files, doc)
        customers = {}
        if "customers.csv" in files:
            customers = {r["customer_id"]: r.get("tier") for r in csv.DictReader(io.StringIO(files["customers.csv"].decode("utf-8")))}
        for i, op in enumerate(ops, start=1):
            if split == "frozen" and spec.family == "inventory" and op.get("col") == "weight":
                op = {**op, "col": "volume"}          # the frozen vocabulary measures litres, not kilograms: same role, other name
            expected, atype, tol = eq.truth(op, rows, customers)
            expected_true, _, _ = eq.truth(op, full, customers)
            text = eq.question_text(op, spec.family) if op.get("col") != "volume" else \
                eq.question_text({**op, "col": "weight"}, spec.family).replace("weight in kilograms", "volume in litres")
            questions.append({"id": f"{tag}-{i}", "tag": tag, "tier": tier, "kind": kind, "case_id": spec.case_id,
                              "family": spec.family, "question": text, "expected": expected,
                              "expected_true": expected_true, "info_lost": expected != expected_true,
                              "answer_type": atype, "tolerance": tol, "op": op, "join": op["op"].startswith("join_")})
    return cases, questions


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for name in PIPELINE_FILES:
        path = os.path.join(HERE, name)
        h.update(name.encode())
        h.update(open(path, "rb").read().replace(b"\r\n", b"\n") if os.path.exists(path) else b"<missing>")
    return h.hexdigest()


def summary(out: dict, arms, questions) -> dict:
    res = {}
    for a in arms:
        row = {"all": sum(out.get((q["id"], a)) == "correct" for q in questions), "n": len(questions)}
        for kind in ("structural", "value"):
            qs = [q for q in questions if q["kind"] == kind]
            row[kind] = f"{sum(out.get((q['id'], a)) == 'correct' for q in qs)}/{len(qs)}"
        res[a] = row
    return res


def run_frozen(arms, marker=None, report_path=None, out=print) -> int:
    """Score the held-out split ONCE (validators_targets.FROZEN_POLICY). A second call refuses: no retune-and-rescore."""
    marker = marker or FROZEN_MARKER
    if os.path.exists(marker):
        prev = json.load(open(marker))
        out(f"REFUSED: the frozen split was already scored on {prev.get('scored_utc')} (code {str(prev.get('code_fingerprint'))[:12]}).")
        out("Policy: scored once, with the final settings; a failure is reported as a failure; no retune-and-rescore.")
        out(f"The result is in {os.path.basename(report_path or FROZEN_REPORT)} and {os.path.basename(marker)}.")
        return 3
    cases, questions = build_split("frozen")
    results, skipped = measure(arms, cases, questions)
    text = report(results, skipped, arms, questions)
    stamp = {"scored_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             "code_fingerprint": code_fingerprint(), "arms": [a for a in arms if a not in skipped], "n_questions": len(questions),
             "summary": summary(results, [a for a in arms if a not in skipped], questions), "skipped": skipped,
             "note": "held-out dialect split; scored once; mechanical solver (the table is scored, not a model)"}
    rp = report_path or FROZEN_REPORT
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(marker, "w", encoding="utf-8") as f:       # written only after a complete run
        json.dump(stamp, f, indent=2)
    out(text)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(KNOWN_ARMS))
    ap.add_argument("--split", default="dev", choices=("dev", "frozen"),
                    help="dev: the tables the validators were written against. frozen: held-out dialects, scored ONCE.")
    args = ap.parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in KNOWN_ARMS]
    if bad:
        print("unknown arm(s):", bad, "known:", KNOWN_ARMS)
        return 2
    if args.split == "frozen":
        return run_frozen(arms)
    cases, questions = eq.build_exp()
    out, skipped = measure(arms, cases, questions)
    print(report(out, skipped, arms, questions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
