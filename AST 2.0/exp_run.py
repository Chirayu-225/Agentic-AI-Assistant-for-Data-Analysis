"""
exp_run.py — runs the arms of the experiment (see exp_rules.py for the pre-registration).

    python exp_run.py --write-prereg                       # once: fixes the question set, data hash, model, rules
    python exp_run.py --dry-run                            # 5 runs on the real model: prints tokens per call, NOT correctness
    python exp_run.py --solver mechanical --pass check     # no model, no quota: plumbing and question sanity
    python exp_run.py --solver llm --pass shakedown1       # the one-day, 1-repeat shakedown (single tier)
    python exp_run.py --solver llm --pass full --tier all --repeats 3 --max-runs 150    # resumable across days

An LLM run refuses to start unless: the working tree is clean (so every run maps to a commit), PREREG.json exists
and matches the regenerated question set and data, and the model matches the pinned one. Every run is appended to
the committed RUNS.jsonl, and the run count is printed before anything else.

The "mechanical" solver is NOT an LLM. It applies the obvious pandas operation to whatever table an arm produces
(no cleverness, no compensation), through the same bytes-in path. It is for testing the harness and checking that
each question can actually be broken by its defect; it says nothing about what a model does.
"""

from __future__ import annotations

import argparse
import ast
import collections
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
import tracemalloc

import clean_score as cs
import exp_questions as eq
import exp_rules
from eval_harness import Question, check_answer, classify_failure

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(HERE, "eval_data", "exp", "PREREG.json")
RUNS_LOG = os.path.join(HERE, "eval_data", "exp", "RUNS.jsonl")
RESULTS_DIR = os.path.join(HERE, "eval_results", "exp")
ARMS = tuple(exp_rules.RULES["arms"])
ANALYSIS_TEMPERATURE = {"first_attempt": 0.1, "repair": 0.0}   # fixed in analysis_service.py
SLEEP = time.sleep           # patched in tests
MAX_PACE_ATTEMPTS = 12       # waits per run before giving up on a run in pace mode


# ── the system under test must not change mid-run ────────────────────────────
SYSTEM_ROOTS = ("analysis_service", "multi_table_service", "clean_adapter_stage0", "cleaning", "file_reader")


def system_files(root: str = HERE) -> list:
    """Every local module the experiment can execute: the import closure of the analysis path, the join path and the
    Stage 0 adapter, found by reading the code (not by running it). Tests, the harness and the benchmark are not in it."""
    todo, seen = list(SYSTEM_ROOTS), set()
    while todo:
        name = todo.pop()
        path = os.path.join(root, name + ".py")
        if name in seen or not os.path.exists(path):
            continue
        seen.add(name)
        for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
            if isinstance(node, ast.Import):
                todo += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                todo.append(node.module.split(".")[0])
    return sorted(n + ".py" for n in seen)


def fingerprint(root: str = HERE) -> str:
    h = hashlib.sha256()
    for f in system_files(root):
        h.update(f.encode())
        h.update(open(os.path.join(root, f), "rb").read())
    return h.hexdigest()[:16]


def make_stamp(commit) -> dict:
    pin = model_pin()
    return {"commit": commit, "model": pin["code_model"], "temperature": pin["analysis_temperature"]["first_attempt"],
            "system_sha": fingerprint()}


def check_continuity(path: str, pass_name: str, current: str, runs: list) -> tuple:
    """A resume must continue the same system. -> (ok, message). Cells stamped with another fingerprint, or unstamped
    cells nobody has attested to, make the pooled number meaningless, so the run refuses to continue."""
    if not os.path.exists(path):
        return True, ""
    cells = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    shas = {c["system_sha"] for c in cells if c.get("system_sha")}
    if shas - {current}:
        return False, (f"REFUSED: cells in this pass were produced by a different version of the system under test "
                       f"({sorted(shas - {current})} vs now {current}). Restore the code, or start a new pass.")
    legacy = [c for c in cells if not c.get("system_sha") and c["outcome"] != "infra_error"]
    attested = {r["system_sha"] for r in runs if r.get("attest") and r.get("pass") == pass_name}
    if legacy and current not in attested:
        return False, (f"REFUSED: {len(legacy)} cell(s) predate per-cell stamping and have not been attested for this system. "
                       f"Run: python exp_run.py --attest --pass {pass_name}")
    return True, ""


def git_changed(c0: str, root: str = HERE) -> list:
    out = subprocess.run(["git", "-C", root, "diff", "--name-only", c0, "HEAD"], capture_output=True, text=True, timeout=60).stdout
    return [os.path.basename(p) for p in out.splitlines() if p.strip()]


def git_diff_lines(c0: str, name: str, root: str = HERE) -> list:
    out = subprocess.run(["git", "-C", root, "diff", c0, "HEAD", "--", name], capture_output=True, text=True, timeout=60).stdout
    return [l for l in out.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---"))]


def attest(pass_name: str, changed=git_changed, diff=git_diff_lines, log_path: str = None, out=print) -> int:
    """Vouches for cells recorded before per-cell stamping, but only after CHECKING git: the system's files must be
    identical to the first session's commit, except llm_provider.py, whose diff must be pure additions (no line removed
    or edited, so no retry/backoff/model/temperature setting can have moved)."""
    log_path = log_path or RUNS_LOG
    runs = cs.read_log(log_path)
    first = next((r for r in runs if r.get("pass") == pass_name and not r.get("attest")), None)
    if not first:
        out(f"No session of pass '{pass_name}' in the log.")
        return 2
    touched = [f for f in changed(first["commit"]) if f in system_files()]
    bad = [f for f in touched if f != "llm_provider.py"]
    extra = []
    if "llm_provider.py" in touched:
        lines = diff(first["commit"], "llm_provider.py")
        removed = [l for l in lines if l.startswith("-")]
        if removed:
            bad.append("llm_provider.py (lines removed or edited)")
        extra = [l for l in lines if l.startswith("+")]
    if bad:
        out(f"REFUSED: the system under test changed since the first session ({first['commit'][:8]}): {bad}")
        return 2
    rec = {"ts": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0).isoformat(), "attest": True, "pass": pass_name,
           "system_sha": fingerprint(), "since_commit": first["commit"],
           "verified": {"system_files_unchanged_except": touched, "llm_provider_pure_additions": len(extra)},
           "covers": "cells of this pass recorded without a system_sha"}
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    out(f"attested: the system files are unchanged since {first['commit'][:8]}" + (f" apart from {len(extra)} added line(s) in llm_provider.py" if extra else "")
        + f"; system_sha {rec['system_sha']}. Commit RUNS.jsonl.")
    return 0


# ── the tables the arms analyse ──────────────────────────────────────────────
def prepare(arm: str, files: dict, doc: dict):
    """-> (bytes or None, why-not, seconds, peak MB) for the table this arm analyses."""
    tracemalloc.start()
    t0 = time.time()
    err = ""
    if arm == "clean":
        data = eq.attainable_csv(files, doc)  # the ceiling: what a perfect cleaner could deliver from the dirty file
    elif arm == "dirty":
        data = files["data.csv"]
    else:
        from clean_adapter_stage0 import cleaned_csv_bytes
        data, err = cleaned_csv_bytes(files["data.csv"])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return data, err, round(time.time() - t0, 3), round(peak / 1e6, 1)


# ── solvers ──────────────────────────────────────────────────────────────────
def llm_solver(data: bytes, q: dict, customers, session: str) -> dict:
    if q["join"]:
        from multi_table_service import run_multi_table_query
        return run_multi_table_query([{"alias": "orders", "csv_bytes": data, "filename": "orders.csv"},
                                      {"alias": "customers", "csv_bytes": customers, "filename": "customers.csv"}],
                                     q["question"], session)
    from analysis_service import run_analysis
    return run_analysis(data, "data.csv", q["question"], session)


def mechanical_solver(data: bytes, q: dict, customers, session: str) -> dict:
    import pandas as pd

    import cleaning
    import file_reader

    def load(b):
        df = file_reader._read_tabular_file_uncached(b, "data.csv")
        return cleaning.rule_based_clean(df)[0]

    err = lambda m: {"success": False, "resultType": "error", "resultData": None, "error": m}
    try:
        df = load(data)
    except Exception as e:  # noqa: BLE001
        return err(f"Could not parse file: {e}")
    op, col, num = q["op"], q["op"].get("col"), lambda c: pd.to_numeric(df[c], errors="coerce")
    try:
        o = op["op"]
        if o == "count_rows":
            v = len(df)
        elif o == "count_distinct":
            v = df[col].nunique()
        elif o == "sum":
            v = num(col).sum()
        elif o == "avg":
            v = num(col).mean()
        elif o == "max":
            v = num(col).max()
        elif o == "count_value":
            v = (df[col] == op["value"]).sum()
        elif o == "share_value":
            v = (df[col] == op["value"]).mean() * 100
        elif o == "count_year":
            v = (pd.to_datetime(df[col], errors="coerce").dt.year == op["year"]).sum()
        elif o == "max_year":
            v = pd.to_datetime(df[col], errors="coerce").dt.year.max()
        elif o == "avg_pct":
            v = num(col).mean() * 100
        elif o == "avg_tenure_years":
            v = (pd.to_datetime(df["end_date"], errors="coerce") - pd.to_datetime(df["start_date"], errors="coerce")).dt.days.mean() / 365.25
        elif o in ("join_count_tier", "join_sum_tier"):
            cust = load(customers)
            m = df.merge(cust[["customer_id", "tier"]], on="customer_id", how="inner")
            m = m[m["tier"] == op["value"]]
            v = len(m) if o == "join_count_tier" else pd.to_numeric(m["amount"], errors="coerce").sum()
        else:
            return err(f"no mechanical rule for {o}")
        return {"success": True, "resultType": "scalar", "resultData": float(v), "error": None}
    except Exception as e:  # noqa: BLE001
        return err(f"mechanical solver failed: {type(e).__name__}: {e}")


SOLVERS = {"llm": llm_solver, "mechanical": mechanical_solver}


# ── one run ──────────────────────────────────────────────────────────────────
def classify(q: dict, result: dict):
    """-> (outcome, actual). Outcomes: correct | wrong | no_answer | error | crash | infra_error."""
    err = str(result.get("error") or "")
    if classify_failure(err) == "infra_error":
        return "infra_error", ""
    if err.startswith(("ARM_CRASH", "Could not parse file")):
        return "crash", ""
    if not result.get("success", result.get("resultType") != "error"):
        return "error", ""
    verdict = check_answer(Question(id=q["id"], question=q["question"], expected=q["expected"], answer_type=q["answer_type"],
                                    tolerance=q["tolerance"]), result)
    if not verdict.comparable:
        return "no_answer", verdict.actual
    return ("correct" if verdict.correct else "wrong"), verdict.actual


def _snap():
    import llm_provider as lp
    return dict(lp.token_stats()), lp.usage_stats().get("calls", 0)


def run_one(q: dict, arm: str, repeat: int, case: tuple, solver_name: str, pass_name: str, stamp: dict = None) -> dict:
    _, files, doc = case
    data, why, prep_s, prep_mb = prepare(arm, files, doc)
    rec = {"ts": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0).isoformat(), "pass": pass_name, "qid": q["id"],
           "tag": q["tag"], "tier": q["tier"], "kind": q["kind"], "arm": arm, "repeat": repeat, "expected": q["expected"],
           "solver": solver_name, "prep_s": prep_s, "prep_peak_mb": prep_mb, "calls": 0, "tokens_prompt": 0, "tokens_completion": 0,
           **(stamp or {})}
    if data is None:
        result = {"success": False, "resultType": "error", "resultData": None, "error": f"ARM_CRASH: {why}"}
        rec["latency_s"] = 0.0
    else:
        if solver_name == "llm":
            import llm_provider as lp
            lp.clear_last_rate_limit()
        t0, (tok0, calls0) = time.time(), (_snap() if solver_name == "llm" else ({"prompt": 0, "completion": 0}, 0))
        try:
            result = SOLVERS[solver_name](data, q, files.get("customers.csv"), f"exp-{pass_name}-{q['id']}-{arm}-r{repeat}")
        except Exception as e:  # noqa: BLE001 — a solver crash is a result
            result = {"success": False, "resultType": "error", "resultData": None, "error": f"{type(e).__name__}: {e}"}
        rec["latency_s"] = round(time.time() - t0, 2)
        if solver_name == "llm":
            tok1, calls1 = _snap()
            rec.update(calls=calls1 - calls0, tokens_prompt=tok1["prompt"] - tok0["prompt"], tokens_completion=tok1["completion"] - tok0["completion"])
    rec["outcome"], rec["actual"] = classify(q, result)
    rec["note"] = bool(result.get("notes") or result.get("cleaningNotes"))
    rec["detail"] = str(result.get("error") or "")[:160] if rec["outcome"] not in ("correct", "wrong") else ""
    return rec


# ── guards, pre-registration, log ────────────────────────────────────────────
def model_pin() -> dict:
    import llm_provider as lp
    return {"code_model": lp._CODE_MODEL, "chat_model": lp._CHAT_MODEL, "analysis_temperature": ANALYSIS_TEMPERATURE}


def prereg_doc(questions: list) -> dict:
    tiers = {t: sum(1 for q in questions if q["tier"] == t) for t in ("single", "compound")}
    return {"written_utc": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0).isoformat(),
            "questions_sha256": eq.questions_sha256(questions), "data_sha256": eq.data_sha256(), "n_questions": tiers,
            "model": model_pin(), "rules": exp_rules.RULES,
            "note": "Committed before any LLM run. The shakedown is not a verdict (see exp_rules.py)."}


def write_prereg(force: bool = False) -> int:
    _, questions = eq.build_exp()
    doc = prereg_doc(questions)
    if os.path.exists(PREREG) and not force:
        old = json.load(open(PREREG, encoding="utf-8"))
        same = all(old.get(k) == doc[k] for k in ("questions_sha256", "data_sha256", "model", "rules"))
        print("PREREG.json already exists and " + ("matches the current question set, data, model and rules." if same else
              "DIFFERS from the current question set / data / model / rules. Refusing to overwrite a pre-registration."))
        return 0 if same else 2
    os.makedirs(os.path.dirname(PREREG), exist_ok=True)
    json.dump(doc, open(PREREG, "w", encoding="utf-8"), indent=1)
    print(f"wrote {PREREG}\n  questions: {doc['n_questions']}  questions_sha256 {doc['questions_sha256'][:16]}  data_sha256 {doc['data_sha256'][:16]}\n"
          f"  model {doc['model']['code_model']}  analysis temperature {doc['model']['analysis_temperature']}\n"
          "  Commit PREREG.json, exp_rules.py and the code before the first LLM run.")
    return 0


def guard_llm(questions: list, git=cs.git_state):
    commit, dirty = git()
    if commit is None or dirty is None:
        return False, ["REFUSED: cannot read git state, so this run could not be tied to a commit."], None
    if dirty:
        return False, [f"REFUSED: the working tree has {dirty} uncommitted change(s). Commit first (a previous run's RUNS.jsonl line counts)."], commit
    if not os.path.exists(PREREG):
        return False, ["REFUSED: PREREG.json does not exist. Run --write-prereg, then commit."], commit
    pre = json.load(open(PREREG, encoding="utf-8"))
    if pre["questions_sha256"] != eq.questions_sha256(questions):
        return False, ["REFUSED: the question set no longer matches PREREG.json."], commit
    if pre["data_sha256"] != eq.data_sha256():
        return False, ["REFUSED: the generated tables no longer match PREREG.json."], commit
    if pre["model"] != model_pin():
        return False, [f"REFUSED: the model or temperature differs from the pinned one ({pre['model']} vs {model_pin()})."], commit
    runs = cs.read_log(RUNS_LOG)
    last = runs[-1] if runs else None
    return True, [f"EXPERIMENT RUN #{len(runs) + 1}  |  last run: {last['ts'] + ' @ ' + last['commit'][:8] + ' (' + last['pass'] + ')' if last else 'never'}  |  "
                  f"model {pre['model']['code_model']}"], commit


def load_done(path: str) -> dict:
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r["outcome"] != "infra_error":   # an infra error is not a result: its cell must be run again
                    done[(r["qid"], r["arm"], r["repeat"])] = r
    return done


def provider_wait(rec: dict) -> float:
    """Seconds to wait before retrying a run the provider refused: its own retry-after (the quota is a refilling
    bucket), else a conservative default."""
    import llm_provider as lp
    info = lp.last_rate_limit()
    if info and info.get("retry_after"):
        return min(float(info["retry_after"]) + 15.0, 3600.0)
    return 900.0 if "daily ai usage limit" in rec["detail"].lower() else 120.0


def run_pass(solver_name, pass_name, tier, repeats, max_runs, path, questions, cases, out=print, pace=False, stamp=None) -> dict:
    """pace=True: when the provider refuses a run, wait for its quota bucket to refill and retry the SAME run
    (same question, arm and repeat: nothing is substituted), so one unattended session can span days.
    A question's arms are a BLOCK: they run together, and a block left unfinished by an earlier session is re-run in
    full on resume, so one question's arms never straddle a pause (the day-confound blocking was meant to remove)."""
    done, n, outcomes, stopped = load_done(path), 0, {}, None
    chosen = [q for q in questions if tier in ("all", q["tier"])]
    blocks = collections.defaultdict(set)
    for (qid, arm, rep) in done:
        blocks[(qid, rep)].add(arm)
    unfinished = sorted(k for k, arms in blocks.items() if len(arms) < len(ARMS))
    if unfinished:
        for key in [k for k in done if (k[0], k[2]) in unfinished]:
            del done[key]
        out(f"  resuming: re-running {len(unfinished)} unfinished question block(s) in full: {', '.join(f'{q} r{r}' for q, r in unfinished)}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tot = {"calls": 0, "tokens_prompt": 0, "tokens_completion": 0}
    try:
        for rep in range(1, repeats + 1):
            for q in chosen:
                todo = [arm for arm in ARMS if (q["id"], arm, rep) not in done]
                if not todo:
                    continue
                if max_runs and n >= max_runs:   # the budget is checked between blocks, never inside one
                    stopped = "max_runs"
                    break
                for arm in todo:
                    attempts = 0
                    while True:
                        rec = run_one(q, arm, rep, cases[q["case_id"]], solver_name, pass_name, stamp)
                        with open(path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec) + "\n")
                        n += 1
                        outcomes[rec["outcome"]] = outcomes.get(rec["outcome"], 0) + 1
                        for k in tot:
                            tot[k] += rec[k]
                        out(f"  {q['id']:<13}{arm:<8}r{rep}  {rec['outcome']}")
                        if rec["outcome"] != "infra_error":
                            break
                        if not pace:
                            if "daily" in rec["detail"].lower():
                                stopped = "daily_quota"
                            break
                        attempts += 1
                        if attempts > MAX_PACE_ATTEMPTS:
                            stopped = "gave_up"
                            break
                        wait = provider_wait(rec)
                        out(f"  provider quota: waiting {wait / 60:.1f} min, then retrying this same run (attempt {attempts})")
                        SLEEP(wait)
                    if stopped:
                        break
                if stopped:
                    break
            if stopped:
                break
    except KeyboardInterrupt:
        stopped = "interrupted"
    return {"runs": n, "outcomes": outcomes, "stopped": stopped, **tot}


def append_log(commit, pass_name, solver_name, tier, repeats, summary):
    rec = {"ts": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0).isoformat(), "commit": commit, "pass": pass_name,
           "solver": solver_name, "tier": tier, "repeats": repeats, "model": model_pin(), **summary}
    os.makedirs(os.path.dirname(RUNS_LOG), exist_ok=True)
    with open(RUNS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ── dry run: tokens per call, never correctness ──────────────────────────────
def dry_run(questions, cases, tpd: int, out=print) -> dict:
    picks = [("footer-1", "dirty"), ("datemix-1", "dirty"), ("sentinel-1", "stage0"), ("outlier-1", "dirty"), ("catvar-1", "clean")]
    qmap = {q["id"]: q for q in questions}
    rows = []
    for qid, arm in picks:
        q = qmap[qid]
        rec = run_one(q, arm, 1, cases[q["case_id"]], "llm", "dryrun")
        rows.append(rec)
        out(f"  {qid:<12}{arm:<8} calls {rec['calls']}  tokens in/out {rec['tokens_prompt']}/{rec['tokens_completion']}  {rec['latency_s']}s"
            + ("" if rec["outcome"] != "infra_error" else "  [infra error: provider]"))
    return dry_summary(rows, tpd)


def dry_summary(rows: list, tpd: int) -> dict:
    ok = [r for r in rows if r["outcome"] != "infra_error" and r["calls"] > 0]
    calls = sum(r["calls"] for r in ok)
    tokens = sum(r["tokens_prompt"] + r["tokens_completion"] for r in ok)
    if not ok or not calls or not tokens:
        return {"usable": False}
    per_call, calls_per_run = tokens / calls, calls / len(ok)
    return {"usable": True, "runs": len(ok), "calls": calls, "tokens_per_call": round(per_call), "calls_per_run": round(calls_per_run, 2),
            "tokens_per_run": round(tokens / len(ok)), "runs_per_day": int(0.9 * tpd / (tokens / len(ok))),
            "recommended_daily_call_limit": int(0.9 * tpd / per_call),
            "shakedown_runs": 40 * 3, "shakedown_tokens": round(40 * 3 * tokens / len(ok))}


def main(argv=None, git=cs.git_state) -> int:
    ap = argparse.ArgumentParser(description="Experiment arms harness")
    ap.add_argument("--solver", choices=sorted(SOLVERS), default="mechanical")
    ap.add_argument("--pass", dest="pass_name", default="check")
    ap.add_argument("--tier", choices=["single", "compound", "all"], default="single")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--write-prereg", action="store_true")
    ap.add_argument("--force-prereg", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--attest", action="store_true", help="vouch (after checking git) for cells recorded before per-cell stamping")
    ap.add_argument("--pace", action="store_true", help="wait out the provider's refilling quota and carry on (one unattended session)")
    ap.add_argument("--tpd", type=int, default=200000, help="the provider's tokens-per-day limit (200000 seen on Groq's free tier)")
    a = ap.parse_args(argv)
    if a.write_prereg:
        return write_prereg(a.force_prereg)
    if a.attest:
        return attest(a.pass_name)
    cases, questions = eq.build_exp()
    if a.dry_run:
        print("DRY RUN: 5 real model runs. Prints tokens and latency only; correctness is neither shown nor stored.")
        s = dry_run(questions, cases, a.tpd)
        if not s["usable"]:
            print("No usable runs (provider error or no token usage reported). Check GROQ_API_KEY and the quota.")
            return 1
        print(f"\ntokens per call: {s['tokens_per_call']}   calls per run: {s['calls_per_run']}   tokens per run: {s['tokens_per_run']}"
              f"\nat {a.tpd:,} tokens/day: about {s['runs_per_day']} runs/day (90% of the quota)"
              f"\nrecommended LLM_DAILY_CALL_LIMIT: {s['recommended_daily_call_limit']}  (the default 400 assumes ~500 tokens per call)"
              f"\nshakedown (40 questions x 3 arms x 1 repeat = {s['shakedown_runs']} runs): about {s['shakedown_tokens']:,} tokens "
              f"= {s['shakedown_tokens'] / a.tpd:.0%} of one day")
        return 0
    commit = None
    if a.solver == "llm":
        ok, lines, commit = guard_llm(questions, git)
        print("\n".join(lines))
        if not ok:
            return 2
        ok, why = check_continuity(os.path.join(RESULTS_DIR, f"{a.pass_name}-{a.solver}.jsonl"), a.pass_name, fingerprint(), cs.read_log(RUNS_LOG))
        if not ok:
            print(why)
            return 2
    else:
        print("MECHANICAL solver: no model is called. This is plumbing and question sanity, not evidence about a model.")
    path = os.path.join(RESULTS_DIR, f"{a.pass_name}-{a.solver}.jsonl")
    summary = run_pass(a.solver, a.pass_name, a.tier, a.repeats, a.max_runs, path, questions, cases, pace=a.pace,
                       stamp=make_stamp(commit) if a.solver == "llm" else None)
    print(f"\n{summary['runs']} runs {summary['outcomes']}" + (f"  STOPPED: {summary['stopped']} (commit the log line, then re-run the same command to resume)" if summary["stopped"] else ""))
    if a.solver == "llm":
        append_log(commit, a.pass_name, a.solver, a.tier, a.repeats, {**summary, "system_sha": fingerprint()})
        print(f"logged to RUNS.jsonl (commit it before the next LLM run); results: {path}")
    else:
        print(f"results: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
