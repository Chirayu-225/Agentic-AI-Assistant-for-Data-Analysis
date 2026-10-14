"""
eval_harness.py — measures how well the assistant answers questions about a
REAL dataset, against ground-truth answers YOU computed independently.

HOW THIS DIFFERS FROM eval.py
    eval.py tests deterministic internals (shortcut detection, cleaning
    rules, sandbox blocking) with no LLM. This harness tests the thing a
    user actually experiences: ask a question in English, get an answer.
    It runs each question through the real analysis pipeline and compares
    the result with an expected answer you worked out by hand (Excel, a
    calculator, a pivot table) — so the score is evidence about the system,
    not about the system agreeing with itself.

WORKFLOW
    1. python eval_harness.py --init-template mydata.csv
           writes questions_template.csv with example rows built from the
           file's real column names.
    2. Fill in `expected` for each question BY HAND, independently of the
       assistant. Delete rows you don't want. Add your own questions.
       (Optional `manual_seconds`: how long the task took you in Excel,
       so the report can compare effort.)
    3. python eval_harness.py --dataset mydata.csv --questions questions.csv
           needs GROQ_API_KEY (it calls the real LLM pipeline).
       or add  --solver shortcut  to run the deterministic, no-LLM
           baseline (answers only questions the keyword shortcuts cover).
    4. Open eval_results/<timestamp>/report.md. Put results.csv in your
       appendix, and the numbers in your report.

QUESTIONS FILE (CSV, or a JSON list of the same fields)
    id              optional label (default Q1, Q2, ...)
    question        the natural-language question, exactly as a user would type it
    expected        the ground-truth answer you computed
    answer_type     number | text | contains | rows | chart
                      number   a single numeric answer (compared within tolerance)
                      text     a single text answer (case/space-insensitive equality)
                      contains expected is "a|b|c": every item must appear in the answer
                      rows     the answer is a table; expected is its row count
                      chart    expected ignored; passes if a chart was produced
    tolerance       relative tolerance for `number` (default 0.01 = 1%)
    category        free label for grouping results (count, aggregate, groupby, ...)
    manual_seconds  optional: how long you took to do it manually

WHAT IT MEASURES
    accuracy              correct / asked, with a 95% Wilson interval
    execution success     the pipeline produced an answer at all, with its own interval
    failure breakdown     wrong answer / no comparable answer / security block /
                          timeout / memory limit / execution error / pipeline (LLM) error / needs LLM
    latency               mean, median, p95, max seconds per question
    consistency (--repeat K)  how often repeated runs of the same question agree —
                          LLM output is non-deterministic, so one run proves little
    effort                total manual time vs total assistant time (if provided)

HONEST LIMITS
    - It is only as good as your ground truth and your question sample. 20
      questions give a wide interval; the report prints it so nobody reads
      "85%" as more certain than it is.
    - Latency is the assistant's compute time, not the user's total time
      (reading, typing, checking). Compare effort with that in mind.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ANSWER_TYPES = ("number", "text", "contains", "rows", "chart")
DEFAULT_TOLERANCE = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Question:
    id: str
    question: str
    expected: str = ""
    answer_type: str = "number"
    tolerance: float = DEFAULT_TOLERANCE
    category: str = ""
    manual_seconds: Optional[float] = None


@dataclass
class QuestionResult:
    id: str
    run: int
    question: str
    category: str
    answer_type: str
    expected: str
    actual: str
    correct: bool
    executed: bool            # the pipeline produced an answer (right or wrong)
    outcome: str              # correct | wrong_answer | no_answer | security_block | timeout | execution_error | pipeline_error | needs_llm | solver_exception
    detail: str               # human-readable reason
    latency_s: float
    result_type: str = ""
    confidence: str = ""
    manual_seconds: Optional[float] = None
    signature: str = ""      # answer VALUES only (no column names / formatting), used for repeat consistency


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION FILES
# ─────────────────────────────────────────────────────────────────────────────

def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return default


def load_questions(path: str) -> list[Question]:
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    else:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

    questions: list[Question] = []
    for i, row in enumerate(rows, start=1):
        row = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
        text = str(row.get("question", "")).strip()
        if not text:
            continue
        atype = str(row.get("answer_type", "") or "number").strip().lower()
        if atype not in ANSWER_TYPES:
            raise ValueError(f"Row {i}: answer_type '{atype}' must be one of {', '.join(ANSWER_TYPES)}")
        expected = str(row.get("expected", "") or "").strip()
        if atype != "chart" and (expected == "" or expected.startswith("<")):
            raise ValueError(
                f"Row {i} ({text[:40]!r}): `expected` is still empty/a placeholder. "
                f"Fill in the ground-truth answer you computed by hand, or delete the row."
            )
        questions.append(Question(
            id=str(row.get("id") or f"Q{len(questions) + 1}").strip(),
            question=text,
            expected=expected,
            answer_type=atype,
            tolerance=_to_float(row.get("tolerance"), DEFAULT_TOLERANCE),
            category=str(row.get("category", "") or "").strip(),
            manual_seconds=_to_float(row.get("manual_seconds")),
        ))
    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


def write_template(dataset_path: str, out_path: str) -> str:
    """Writes a starter questions file built from the dataset's real columns.
    `expected` is deliberately left as a placeholder: the point is for YOU to
    compute it independently."""
    import pandas as pd
    from file_reader import read_tabular_file
    with open(dataset_path, "rb") as f:
        df = read_tabular_file(f.read(), os.path.basename(dataset_path))
    def _is_id_like(c) -> bool:
        # Identifier columns (order_id, ...) make meaningless "total"/"average"
        # questions, so prefer real measures when picking example columns.
        name = str(c).lower()
        return name == "id" or name.endswith("_id") or name.endswith(" id") or \
            (len(df) > 1 and df[c].nunique(dropna=True) == len(df))

    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in numeric if not _is_id_like(c)] or numeric
    categorical = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num = numeric[0] if numeric else "<numeric column>"
    cat = categorical[0] if categorical else "<category column>"

    rows = [
        ("Q1", "How many rows are in the dataset?", "<your answer>", "number", "0", "count"),
        ("Q2", f"What is the total {num}?", "<your answer>", "number", "0.01", "aggregate"),
        ("Q3", f"What is the average {num}?", "<your answer>", "number", "0.01", "aggregate"),
        ("Q4", f"Which {cat} has the highest total {num}?", "<your answer>", "text", "", "groupby"),
        ("Q5", f"List the top 3 {cat} by total {num}", "<a|b|c>", "contains", "", "ranking"),
        ("Q6", f"How many distinct {cat} values are there?", "<your answer>", "number", "0", "count"),
        ("Q7", f"Plot {num} by {cat} as a bar chart", "", "chart", "", "chart"),
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "expected", "answer_type", "tolerance", "category", "manual_seconds"])
        for r in rows:
            w.writerow([*r, ""])
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER EXTRACTION AND CHECKING
# ─────────────────────────────────────────────────────────────────────────────

def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def _cells(result: dict) -> list:
    """Every cell of a table result, flattened (scalar -> one cell)."""
    data = result.get("resultData")
    if result.get("resultType") == "table" and isinstance(data, list):
        return [v for row in data if isinstance(row, dict) for v in row.values()]
    if result.get("resultType") == "scalar":
        return [data]
    return []


def _table_rows(result: dict) -> Optional[int]:
    if result.get("resultType") != "table":
        return None
    total = result.get("totalRowCount")
    if isinstance(total, int):
        return total
    data = result.get("resultData")
    return len(data) if isinstance(data, list) else None


def extract_number(result: dict) -> tuple[Optional[float], str]:
    """A single numeric answer: a scalar, or a one-row table with exactly one
    numeric cell. Anything more ambiguous returns None rather than guessing."""
    rtype = result.get("resultType")
    data = result.get("resultData")
    if rtype == "scalar":
        if _is_number(data):
            return float(data), ""
        parsed = _to_float(data)
        return (parsed, "") if parsed is not None else (None, f"scalar answer {data!r} is not numeric")
    if rtype == "table" and isinstance(data, list):
        if len(data) != 1:
            return None, f"answer is a {len(data)}-row table, expected a single value"
        nums = [v for v in data[0].values() if _is_number(v)]
        if len(nums) == 1:
            return float(nums[0]), ""
        return None, f"one-row table has {len(nums)} numeric cells, expected exactly 1"
    return None, f"answer type '{rtype}' has no numeric value"


def _norm_text(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v).strip().lower())


@dataclass
class Verdict:
    correct: bool
    actual: str
    detail: str = ""
    comparable: bool = True   # False = the answer had no value we could compare at all


def check_answer(q: Question, result: dict) -> Verdict:
    """Compares one pipeline result with the ground truth for `q`."""
    rtype = result.get("resultType")

    if q.answer_type == "chart":
        ok = rtype == "chart"
        return Verdict(ok, f"[{rtype}]", "" if ok else f"expected a chart, got '{rtype}'", comparable=ok)

    if q.answer_type == "number":
        expected = _to_float(q.expected)
        if expected is None:
            return Verdict(False, "", f"expected value {q.expected!r} is not a number", comparable=False)
        actual, why = extract_number(result)
        if actual is None:
            return Verdict(False, _preview(result), why, comparable=False)
        tol = q.tolerance if q.tolerance is not None else DEFAULT_TOLERANCE
        err = abs(actual - expected)
        # Relative tolerance; when the truth is 0 a relative tolerance is
        # meaningless, so the tolerance is applied as an absolute amount.
        ok = err <= tol * abs(expected) + 1e-12 if expected != 0 else err <= max(tol, 1e-9)
        detail = "" if ok else f"expected {_fmt_num(expected)}, got {_fmt_num(actual)} (tolerance {tol:g})"
        return Verdict(ok, _fmt_num(actual), detail)

    if q.answer_type == "rows":
        n = _table_rows(result)
        expected = _to_float(q.expected)
        if n is None:
            return Verdict(False, _preview(result), f"answer type '{rtype}' is not a table", comparable=False)
        ok = expected is not None and n == int(expected)
        return Verdict(ok, f"{n} rows", "" if ok else f"expected {q.expected} rows, got {n}")

    cells = _cells(result)
    if not cells:
        return Verdict(False, _preview(result), f"answer type '{rtype}' has no readable values", comparable=False)

    if q.answer_type == "text":
        # Prefer non-numeric cells (the label), fall back to everything.
        candidates = [c for c in cells if not _is_number(c)] or cells
        ok = any(_norm_text(c) == _norm_text(q.expected) for c in candidates)
        return Verdict(ok, _preview(result), "" if ok else f"expected {q.expected!r}, got {_preview(result)}")

    # contains
    haystack = _norm_text(" | ".join(str(c) for c in cells))
    wanted = [w.strip() for w in q.expected.split("|") if w.strip()]
    missing = [w for w in wanted if _norm_text(w) not in haystack]
    return Verdict(not missing, _preview(result), "" if not missing else f"missing from answer: {', '.join(missing)}")


def _fmt_num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.6g}"


def _preview(result: dict, limit: int = 120) -> str:
    rtype = result.get("resultType")
    data = result.get("resultData")
    if rtype == "scalar":
        s = str(data)
    elif rtype == "table" and isinstance(data, list):
        s = f"table[{len(data)} rows] " + json.dumps(data[:2], default=str)
    else:
        s = f"[{rtype}]"
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _norm_cell(v: Any) -> str:
    if _is_number(v):
        return f"{round(float(v), 4):g}"
    parsed = _to_float(v) if isinstance(v, str) else None
    if parsed is not None:
        return f"{round(parsed, 4):g}"
    return _norm_text(v)


def _signature(result: dict) -> str:
    """What the answer SAYS, ignoring how it is labelled: two runs that return
    [{"region": "South", "units": 39}] and [{"region": "South", "total_units": 39}]
    gave the same answer, so they must not count as inconsistent."""
    rtype = result.get("resultType")
    data = result.get("resultData")
    if rtype == "scalar":
        return _norm_cell(data)
    if rtype == "table" and isinstance(data, list):
        rows = [[_norm_cell(v) for v in row.values()] for row in data[:50] if isinstance(row, dict)]
        return json.dumps(rows)
    return f"[{rtype}]"


def classify_failure(error: str) -> str:
    e = (error or "").lower()
    if e.startswith("security block"):
        return "security_block"
    if "no shortcut" in e:
        return "needs_llm"
    if "memory limit" in e:
        return "memory_limit"
    if "execution limit" in e or "timed out" in e or "timeout" in e:
        return "timeout"
    if e.startswith(("analysis pipeline failed", "large-file analysis pipeline failed")) or "llm" in e or "api" in e:
        return "pipeline_error"
    return "execution_error"


# ─────────────────────────────────────────────────────────────────────────────
# SOLVERS — how a question gets answered
# ─────────────────────────────────────────────────────────────────────────────

Solver = Callable[[bytes, str, str, str], dict]


def llm_solver(file_bytes: bytes, filename: str, question: str, session_id: str) -> dict:
    """The real pipeline, exactly as /analyze runs it. Needs GROQ_API_KEY."""
    from analysis_service import run_analysis
    return run_analysis(file_bytes, filename, question, session_id)


def shortcut_solver(file_bytes: bytes, filename: str, question: str, session_id: str) -> dict:
    """No-LLM baseline: parse + rule-clean + the keyword shortcuts + the real
    sandbox. Answers only what query_engine.detect_shortcut recognises, so it
    works offline and shows how much of the score the LLM actually adds."""
    from analysis_service import _shape_result
    from cleaning import rule_based_clean
    from file_reader import read_tabular_file
    from query_engine import detect_shortcut
    from sandbox_limits import run_sandboxed

    df, _ = rule_based_clean(read_tabular_file(file_bytes, filename))
    code = detect_shortcut(question, df)
    if code is None:
        return {"success": False, "resultType": "error", "resultData": None,
                "error": "No shortcut matched this question (needs the LLM)"}
    sbr = run_sandboxed(code, df, timeout_seconds=25)
    if not sbr.success:
        return {"success": False, "resultType": "error", "resultData": None, "error": sbr.error}
    shaped = _shape_result(sbr.value)
    return {"success": shaped["resultType"] != "error", **shaped}


SOLVERS: dict[str, Solver] = {"llm": llm_solver, "shortcut": shortcut_solver}


# ─────────────────────────────────────────────────────────────────────────────
# RUNNING
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(
    file_bytes: bytes,
    filename: str,
    questions: list[Question],
    solver: Solver = llm_solver,
    repeat: int = 1,
    session_id: str = "eval",
    on_result: Optional[Callable[[QuestionResult], None]] = None,
) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    for q in questions:
        for run in range(1, repeat + 1):
            t0 = time.perf_counter()
            try:
                # A UNIQUE session id per (question, run): llm_provider caches
                # generated code keyed on (model, prompt, session_id), so runs
                # sharing a session id would replay the first run's code and
                # make --repeat consistency trivially 100% without ever
                # asking the model again.
                out = solver(file_bytes, filename, q.question, f"{session_id}-{q.id}-run{run}")
                exc = ""
            except Exception as e:  # a solver crash is a result, not a harness crash
                out, exc = {}, f"{type(e).__name__}: {e}"
            latency = time.perf_counter() - t0

            if exc:
                res = QuestionResult(q.id, run, q.question, q.category, q.answer_type, q.expected, "", False, False,
                                     "solver_exception", exc, latency, manual_seconds=q.manual_seconds)
            elif not out.get("success"):
                err = out.get("error") or "unknown error"
                res = QuestionResult(q.id, run, q.question, q.category, q.answer_type, q.expected, "", False, False,
                                     classify_failure(err), err[:300], latency,
                                     out.get("resultType", ""), out.get("confidence", ""), q.manual_seconds)
            else:
                v = check_answer(q, out)
                outcome = "correct" if v.correct else ("wrong_answer" if v.comparable else "no_answer")
                res = QuestionResult(q.id, run, q.question, q.category, q.answer_type, q.expected, v.actual, v.correct,
                                     True, outcome, v.detail, latency, out.get("resultType", ""),
                                     out.get("confidence", ""), q.manual_seconds)
            if res.executed:
                res.signature = _signature(out)
            results.append(res)
            if on_result:
                on_result(res)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS AND REPORT
# ─────────────────────────────────────────────────────────────────────────────

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion — honest about small samples,
    unlike the naive k/n +/- margin, which can exceed [0, 1]."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def summarize(results: list[QuestionResult], repeat: int = 1) -> dict:
    n = len(results)
    correct = sum(r.correct for r in results)
    executed = sum(r.executed for r in results)
    lat = sorted(r.latency_s for r in results)
    acc_lo, acc_hi = wilson_interval(correct, n)
    ex_lo, ex_hi = wilson_interval(executed, n)

    outcomes: dict[str, int] = {}
    for r in results:
        outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1

    by_cat: dict[str, dict] = {}
    for r in results:
        c = by_cat.setdefault(r.category or "(none)", {"n": 0, "correct": 0})
        c["n"] += 1
        c["correct"] += int(r.correct)
    for c in by_cat.values():
        c["accuracy"] = round(c["correct"] / c["n"], 3)

    consistency = None
    if repeat > 1:
        per_q: dict[str, list[str]] = {}
        for r in results:
            per_q.setdefault(r.id, []).append((r.signature or r.actual) if r.executed else f"<{r.outcome}>")
        agree = [max(vals.count(v) for v in set(vals)) / len(vals) for vals in per_q.values()]
        consistency = round(statistics.mean(agree), 3) if agree else None

    effort = None
    paired = [r for r in results if r.manual_seconds and r.run == 1]
    if paired:
        manual = sum(r.manual_seconds for r in paired)
        system = sum(r.latency_s for r in paired)
        effort = {
            "questionsWithManualTime": len(paired),
            "manualSecondsTotal": round(manual, 1),
            "assistantSecondsTotal": round(system, 1),
            "speedup": round(manual / system, 2) if system > 0 else None,
        }

    return {
        "questions": len({r.id for r in results}),
        "runs": n,
        "repeat": repeat,
        "accuracy": round(correct / n, 3) if n else 0,
        "accuracy95": [round(acc_lo, 3), round(acc_hi, 3)],
        "executionSuccess": round(executed / n, 3) if n else 0,
        "executionSuccess95": [round(ex_lo, 3), round(ex_hi, 3)],
        "outcomes": dict(sorted(outcomes.items(), key=lambda kv: -kv[1])),
        "latencySeconds": {
            "mean": round(statistics.mean(lat), 2) if lat else 0,
            "median": round(statistics.median(lat), 2) if lat else 0,
            "p95": round(_percentile(lat, 0.95), 2),
            "max": round(max(lat), 2) if lat else 0,
        },
        "byCategory": by_cat,
        "consistency": consistency,
        "effort": effort,
    }


def render_report(summary: dict, results: list[QuestionResult], meta: dict) -> str:
    pct = lambda x: f"{x * 100:.1f}%"
    lo, hi = summary["accuracy95"]
    elo, ehi = summary["executionSuccess95"]
    L = [
        "# Evaluation report", "",
        f"- Dataset: `{meta.get('dataset', '?')}`",
        f"- Solver: `{meta.get('solver', '?')}`   Generated: {meta.get('generated', '?')}",
        f"- Questions: {summary['questions']}   Runs: {summary['runs']}   Repeats per question: {summary['repeat']}", "",
        "## Headline", "",
        f"- **Accuracy: {pct(summary['accuracy'])}** (95% interval {pct(lo)} – {pct(hi)})",
        f"- Execution success (the assistant produced an answer): {pct(summary['executionSuccess'])} "
        f"(95% interval {pct(elo)} – {pct(ehi)})",
        f"- Latency per question: mean {summary['latencySeconds']['mean']}s, median {summary['latencySeconds']['median']}s, "
        f"p95 {summary['latencySeconds']['p95']}s, max {summary['latencySeconds']['max']}s",
    ]
    if summary["consistency"] is not None:
        L.append(f"- Consistency across repeats: {pct(summary['consistency'])} "
                 f"(how often repeated runs of the same question gave the same answer)")
    if summary["effort"]:
        e = summary["effort"]
        L.append(f"- Effort: {e['manualSecondsTotal']}s by hand vs {e['assistantSecondsTotal']}s assistant compute "
                 f"over {e['questionsWithManualTime']} questions"
                 + (f" (about {e['speedup']}x)" if e["speedup"] else "")
                 + ". Assistant time excludes typing and checking the answer.")
    L += ["", "## Outcomes", "", "| Outcome | Runs |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in summary["outcomes"].items()]
    L += ["", "## By category", "", "| Category | Runs | Correct | Accuracy |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['correct']} | {pct(v['accuracy'])} |" for k, v in summary["byCategory"].items()]
    bad = [r for r in results if not r.correct]
    L += ["", "## Failures", ""]
    if not bad:
        L.append("None.")
    else:
        L += ["| ID | Run | Question | Outcome | Detail |", "|---|---|---|---|---|"]
        for r in bad:
            L.append(f"| {r.id} | {r.run} | {r.question[:60]} | {r.outcome} | {r.detail[:110].replace('|', '/')} |")
    L += ["", "## Reading this honestly", "",
          f"- With {summary['questions']} questions the accuracy interval is wide ({pct(lo)} – {pct(hi)}); "
          "more questions narrow it.",
          "- Ground truth was supplied by the person running the evaluation. Errors in it count against the assistant.",
          "- A non-deterministic LLM can answer the same question differently on another run; use --repeat to measure that.",
          ""]
    return "\n".join(L)


def write_outputs(results: list[QuestionResult], summary: dict, meta: dict, out_root: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(out_root, stamp)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else ["id"])
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary}, f, indent=2)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(render_report(summary, results, meta))
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the assistant against ground-truth answers.")
    ap.add_argument("--dataset", help="CSV/XLSX file to ask questions about")
    ap.add_argument("--questions", help="questions CSV/JSON (see module docstring)")
    ap.add_argument("--init-template", metavar="DATASET", help="write questions_template.csv for this dataset and exit")
    ap.add_argument("--solver", choices=sorted(SOLVERS), default="llm")
    ap.add_argument("--repeat", type=int, default=1, help="run each question K times to measure consistency")
    ap.add_argument("--limit", type=int, default=0, help="only run the first N questions")
    ap.add_argument("--out", default="eval_results")
    args = ap.parse_args(argv)

    if args.init_template:
        path = write_template(args.init_template, "questions_template.csv")
        print(f"Wrote {path}. Fill in `expected` by hand (independently of the assistant), then run:\n"
              f"  python eval_harness.py --dataset {args.init_template} --questions {path}")
        return 0

    if not args.dataset or not args.questions:
        ap.error("--dataset and --questions are required (or use --init-template DATASET)")
    if args.solver == "llm" and not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set — the llm solver calls the real model. "
              "Set it, or try --solver shortcut for the offline baseline.", file=sys.stderr)
        return 2

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    with open(args.dataset, "rb") as f:
        data = f.read()
    filename = os.path.basename(args.dataset)

    print(f"Evaluating {len(questions)} question(s) x {args.repeat} run(s) on {filename} with solver '{args.solver}'\n")

    def show(r: QuestionResult) -> None:
        mark = "OK  " if r.correct else "FAIL"
        extra = "" if r.correct else f"  <- {r.outcome}: {r.detail[:70]}"
        print(f"[{mark}] {r.id}#{r.run} {r.latency_s:5.1f}s  {r.question[:55]}{extra}", flush=True)

    results = run_eval(data, filename, questions, SOLVERS[args.solver], args.repeat,
                       session_id=f"eval-{int(time.time())}", on_result=show)
    summary = summarize(results, args.repeat)
    meta = {"dataset": filename, "solver": args.solver,
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    out_dir = write_outputs(results, summary, meta, args.out)

    lo, hi = summary["accuracy95"]
    print(f"\nAccuracy {summary['accuracy'] * 100:.1f}% (95% interval {lo * 100:.1f}–{hi * 100:.1f}%), "
          f"execution success {summary['executionSuccess'] * 100:.1f}%, "
          f"median latency {summary['latencySeconds']['median']}s")
    if summary.get("consistency") is not None:
        print(f"Repeat consistency {summary['consistency'] * 100:.1f}% "
              f"(mean share of runs per question that gave the same answer)")
    print(f"Report: {os.path.join(out_dir, 'report.md')}")
    return 0 if summary["accuracy"] == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
