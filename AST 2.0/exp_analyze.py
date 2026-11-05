"""
exp_analyze.py — turns experiment results into the pre-registered tables and (for a full run only) the verdicts.

    python exp_analyze.py eval_results/exp/shakedown1-llm.jsonl
    python exp_analyze.py eval_results/exp/full-llm.jsonl --decide

Reading order of the report is deliberate: tokens per call, the per-defect table with n per cell and silent-wrong and
crash counts per arm, the questions the CLEAN arm itself got wrong (is the question set or the model the weak link?),
and only then the headroom. A shakedown (one repeat, or a pass named shakedown*) never evaluates a decision rule.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

import exp_questions as eq
import exp_rules

R = exp_rules.RULES
ARMS = R["arms"]


def load_records(paths) -> list:
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out += [json.loads(l) for l in f if l.strip()]
    return out


def collapse(records: list):
    """Repeats -> one verdict per (question, arm). Infra errors are excluded and counted, never substituted."""
    by, infra = defaultdict(list), Counter()
    for r in records:
        if r["outcome"] == "infra_error":
            infra[(r["qid"], r["arm"])] += 1
        else:
            by[(r["qid"], r["arm"])].append(r)
    maj = {}
    for k, rs in by.items():
        c = Counter(r["outcome"] for r in rs)
        maj[k] = {"n": len(rs), "correct": 2 * c["correct"] > len(rs), "crash": 2 * c["crash"] > len(rs),
                  "silent": sum(1 for r in rs if r["outcome"] == "wrong" and not r["note"]), "crashes": c["crash"],
                  "actual": rs[0]["actual"]}
    return maj, infra


def bootstrap(d: list):
    rng = random.Random(R["headroom"]["seed"])
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(R["headroom"]["bootstrap"]))
    return 100 * means[int(0.025 * len(means))], 100 * means[int(0.975 * len(means)) - 1]


def headroom(questions, maj, tier, kind="value") -> dict:
    """Over questions the clean arm answers correctly: the share the dirty arm gets wrong, with a paired
    bootstrap interval over questions."""
    elig = [q for q in questions if q["tier"] == tier and q["kind"] == kind and maj.get((q["id"], "clean"), {}).get("correct")
            and (q["id"], "dirty") in maj]
    if not elig:
        return {"n": 0, "H": None, "lo": None, "hi": None}
    d = [0 if maj[(q["id"], "dirty")]["correct"] else 1 for q in elig]
    lo, hi = bootstrap(d)
    return {"n": len(d), "H": 100 * sum(d) / len(d), "lo": lo, "hi": hi}


def decide(single: dict, compound: dict) -> str:
    H, lo = single["H"], single["lo"]
    if H is None:
        return "NO VERDICT: no eligible single-tier questions"
    if H >= R["headroom"]["min_points"] and lo > 0:
        return "BUILD the validators (rank by per-type headroom)"
    if H < R["headroom"]["min_points"] and lo <= 0:
        c = compound
        if c["H"] is not None and c["H"] < R["headroom"]["min_points"] and c["lo"] <= 0:
            return "DEPRIORITIZE validators and the review screen; reposition cleaning as transparency (compound tier agrees)"
        return "INCONCLUSIVE: the single tier says deprioritize but the compound tier does not agree (or is missing); add questions"
    return "ADD repeats or questions before deciding"


def stage0_wiring(questions, maj) -> dict:
    structural = set(R["structural_tags"])
    both = lambda q: (q["id"], "dirty") in maj and (q["id"], "stage0") in maj
    per_tag, ok_struct = {}, True
    for tag in structural:
        qs = [q for q in questions if q["tag"] == tag and both(q)]
        s0 = sum(maj[(q["id"], "stage0")]["correct"] for q in qs)
        dr = sum(maj[(q["id"], "dirty")]["correct"] for q in qs)
        per_tag[tag] = (s0, dr, len(qs))
        ok_struct &= s0 >= dr
    new_crashes = [q["id"] for q in questions if both(q) and not maj[(q["id"], "dirty")]["crash"] and maj[(q["id"], "stage0")]["crash"]]
    others = [q for q in questions if q["tag"] not in structural and both(q)]
    worse = sum(1 for q in others if maj[(q["id"], "dirty")]["correct"] and not maj[(q["id"], "stage0")]["correct"])
    better = sum(1 for q in others if not maj[(q["id"], "dirty")]["correct"] and maj[(q["id"], "stage0")]["correct"])
    net_worse = worse - better
    ok = ok_struct and not new_crashes and net_worse <= R["stage0_wiring"]["non_structural_net_worse_max"]
    return {"structural_per_tag": per_tag, "structural_ok": ok_struct, "new_crashes": new_crashes, "net_worse": net_worse,
            "verdict": "WIRE Stage 0 in (and retire the old reader's CSV path)" if ok else "DO NOT wire Stage 0 in yet"}


def gap_closed(questions, maj, kind, tier=None):
    qs = [q for q in questions if q["kind"] == kind and (tier is None or q["tier"] == tier) and all((q["id"], a) in maj for a in ARMS)]
    if not qs:
        return None
    acc = {a: 100 * sum(maj[(q["id"], a)]["correct"] for q in qs) / len(qs) for a in ARMS}
    gap = acc["clean"] - acc["dirty"]
    return {"n": len(qs), "acc": acc, "closed": (acc["stage0"] - acc["dirty"]) / gap if gap >= R["gap_closed_min_clean_minus_dirty"] else None}


def is_shakedown(records: list) -> bool:
    if any(r["pass"].startswith("shakedown") for r in records):
        return True
    return max((r["repeat"] for r in records), default=0) < R["full_run"]["repeats"] or not any(r["tier"] == "compound" for r in records)


def per_tag_table(questions, maj, infra) -> list:
    L = [f"{'defect (tag)':<12}{'kind':<11}{'tier':<9}{'n':>3}  " + "".join(f"| {a:<8}ok  silent  crash  infra  " for a in ARMS)]
    for tag in dict.fromkeys(q["tag"] for q in questions):
        qs = [q for q in questions if q["tag"] == tag]
        row = f"{tag:<12}{qs[0]['kind']:<11}{qs[0]['tier']:<9}{len(qs):>3}  "
        for a in ARMS:
            ev = [maj[(q["id"], a)] for q in qs if (q["id"], a) in maj]
            row += (f"| {'':<8}{sum(m['correct'] for m in ev):>2}/{len(ev):<2} {sum(1 for m in ev if m['silent'] * 2 > m['n']):>4}  "
                    f"{sum(m['crash'] for m in ev):>5}  {sum(infra[(q['id'], a)] for q in qs):>5}  ")
        L.append(row)
    return L


def report(records: list, questions: list, decide_flag: bool = False) -> str:
    maj, infra = collapse(records)
    shake = is_shakedown(records)
    mech = all(r["solver"] == "mechanical" for r in records)
    L = []
    if mech:
        L.append("MECHANICAL SOLVER (not an LLM): plumbing and question sanity only. It says nothing about what a model does.\n")
    if shake:
        L.append("SHAKEDOWN: one repeat cannot support a decision, so no decision rule is evaluated and this must not be quoted as the result.\n")
    llm = [r for r in records if r["solver"] == "llm"]
    calls = sum(r["calls"] for r in llm)
    if llm and calls:
        tok = sum(r["tokens_prompt"] + r["tokens_completion"] for r in llm)
        L.append(f"1. tokens per call: {tok / calls:.0f}  ({calls} calls over {len(llm)} runs = {calls / len(llm):.2f} calls/run, {tok / len(llm):.0f} tokens/run)")
    else:
        L.append("1. tokens per call: n/a (no model calls in these records)")
    L.append(f"   runs: {len(records)}  " + str(dict(Counter(r['outcome'] for r in records))) + "  (infra errors are excluded from every rate and listed per cell)")
    L += ["", "2. per defect type (cells are correct/n questions; silent = wrong answers with no note; crash = the file could not be read)"]
    L += ["   " + x for x in per_tag_table(questions, maj, infra)]
    L += ["", "3. questions the CLEAN arm itself got wrong or could not answer (is the question set or the model the weak link?)"]
    bad = [q for q in questions if (q["id"], "clean") in maj and not maj[(q["id"], "clean")]["correct"]]
    missing = [q for q in questions if (q["id"], "clean") not in maj]
    L += [f"   {q['id']:<12}expected {q['expected']:<14}got {maj[(q['id'], 'clean')]['actual']!r:<14}{q['question'][:60]}" for q in bad] or ["   none"]
    if missing:
        L.append(f"   (no clean-arm result for {len(missing)} question(s): not run, or infra errors)")
    L += ["", "4. headroom, conditioned on questions the clean arm got right (value-level questions)"]
    h_single, h_comp = headroom(questions, maj, "single"), headroom(questions, maj, "compound")
    for name, h in (("single tier", h_single), ("compound tier", h_comp)):
        L.append(f"   {name:<14} " + ("n/a" if h["H"] is None else f"{h['H']:.0f} points over {h['n']} questions, 95% interval [{h['lo']:.0f}, {h['hi']:.0f}]"))
    for kind in ("structural", "value"):
        g = gap_closed(questions, maj, kind)
        if g:
            L.append(f"   {kind:<11} accuracy " + "  ".join(f"{a} {g['acc'][a]:.0f}%" for a in ARMS) + f"  (n={g['n']})  " +
                     ("gap closed by stage0: " + f"{g['closed']:.0%}" if g["closed"] is not None else "gap closed: not reported (clean - dirty < 20 points)"))
    s0 = stage0_wiring(questions, maj)
    L += ["", "5. Stage 0 vs dirty", "   structural (stage0 ok / dirty ok / n): " + ", ".join(f"{t} {v[0]}/{v[1]}/{v[2]}" for t, v in sorted(s0["structural_per_tag"].items())),
          f"   new crashes: {s0['new_crashes'] or 'none'}   non-structural questions net worse: {s0['net_worse']}"]
    if decide_flag and not shake:
        L += ["", "DECISION (pre-registered rules, full run)", f"   validators: {decide(h_single, h_comp)}", f"   stage 0:    {s0['verdict']}"]
    elif decide_flag:
        L += ["", "DECISION: not evaluated (shakedown)."]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--decide", action="store_true")
    a = ap.parse_args(argv)
    _, questions = eq.build_exp()
    print(report(load_records(a.files), questions, a.decide))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
