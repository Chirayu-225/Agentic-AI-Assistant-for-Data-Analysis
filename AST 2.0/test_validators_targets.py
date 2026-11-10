"""test_validators_targets.py -- keeps validators_targets.py honest: the bars are hashed (a tripwire against editing them
after seeing results), names must exist in the benchmark, and the allowlist can never contain a destructive operation."""
import hashlib
import json
import sys

import clean_bench as cb
import validators_targets as vt

FAIL = []


def check(name, cond, detail=""):
    print(("[OK]   " if cond else "[FAIL] ") + name + ("" if cond else f"  -- {detail}"))
    if not cond:
        FAIL.append(name)


# the bars are fixed. If you edit TARGETS, this constant must change in the SAME commit, so the edit is visible.
EXPECTED_BARS_SHA = "1e075ff681d0fb8a"
sha = hashlib.sha256(json.dumps(vt.TARGETS, sort_keys=True).encode()).hexdigest()[:16]
check("the bars match the committed fingerprint (no silent edits)", sha == EXPECTED_BARS_SHA, f"now {sha}")

rates = list(vt.TARGETS["min_recall"].values()) + [vt.TARGETS["max_decoy_hit_rate"], vt.TARGETS["max_decoy_hit_rate_per_type"],
                                                    vt.TARGETS["max_control_cells_flagged"], vt.TARGETS["min_top10_coverage"],
                                                    vt.TARGETS["min_routed_as_expected"]]
check("every rate is a proportion in [0, 1]", all(0 <= r <= 1 for r in rates))
check("the damage bar is a hard zero", vt.TARGETS["max_auto_cells_corrupted"] == 0)
check("at most one LLM call per table, and none when nothing is ambiguous",
      vt.TARGETS["max_llm_calls_per_table"] == 1 and vt.TARGETS["max_llm_calls_when_nothing_ambiguous"] == 0)

existing = {d.name if hasattr(d, "name") else d["name"] for d in cb.DEFECTS}
NEW = {"format_drift"}
measured = set(vt.TARGETS["min_recall"])
check("every measured defect already exists in the benchmark, except the planned new one", (measured - NEW) <= existing, sorted((measured - NEW) - existing))
check("the planned new defect does not collide with an existing one", not (NEW & existing), sorted(NEW & existing))
check("precedence has no duplicates and covers every measured defect",
      len(set(vt.PRECEDENCE)) == len(vt.PRECEDENCE) and measured <= set(vt.PRECEDENCE), vt.PRECEDENCE)

DESTRUCTIVE = {"blank", "drop", "quarantine", "map_values", "cast", "parse_date", "clip", "convert_unit", "replace", "overwrite"}
check("the auto-apply allowlist contains no destructive operation", not (set(vt.AUTO_APPLY_ALLOWLIST) & DESTRUCTIVE), vt.AUTO_APPLY_ALLOWLIST)
check("the allowlist is exactly trim + a normalized COPY (the source value is never overwritten)",
      set(vt.AUTO_APPLY_ALLOWLIST) == {"trim_whitespace", "add_normalized_copy"})

k = vt.SHAPE_KNOBS
check("drift and variance zones do not overlap (a 90% shape leaves at most 10% to a second shape)",
      k["drift_dominant_min"] + k["variance_second_shape_min"] <= 1.0 + 1e-9)
check("the knob log starts empty, so every later change is an explicit entry", vt.KNOB_LOG == [])
required = {"knob", "from", "to", "reason", "dev_before", "dev_after", "date"}
check("any knob-log entry carries every required field", all(required <= set(e) for e in vt.KNOB_LOG))
check("the invariants are listed (8, including the fixed-width trimming edge)", len(vt.ROUTER_INVARIANTS) == 8)

# ── Wilson arithmetic and the gates ──
lo, hi = vt.wilson(100, 100)
check("wilson: 100 of 100 has a lower bound of about 0.963", abs(lo - 0.9630) < 0.001 and hi == 1.0, (lo, hi))
check("wilson: zero hits in 73 trials has an upper bound of about 5%", abs(vt.wilson(0, 73)[1] - 0.0499) < 0.001, vt.wilson(0, 73))
check("wilson: zero hits in 517 trials is well under 2%", vt.wilson(0, 517)[1] < 0.01)
check("wilson: an empty sample says nothing (0 to 1)", vt.wilson(0, 0) == (0.0, 1.0))
check("a false-alarm type with too few columns is NOT GATED even at zero hits", vt.false_alarm_status(0, 27, 0.05) == "NOT_GATED" and vt.false_alarm_status(0, 15, 0.05) == "NOT_GATED")
check("a type with 144 decoys passes at 2 hits and fails at 3 (upper bound vs a 5% bar)",
      vt.false_alarm_status(2, 144, 0.05) == "PASS" and vt.false_alarm_status(3, 144, 0.05) == "FAIL")
check("a type with 77 decoys tolerates zero false alarms only", vt.false_alarm_status(0, 77, 0.05) == "PASS" and vt.false_alarm_status(1, 77, 0.05) == "FAIL")
check("the overall 2% bar on 517 decoys passes at 3 hits and fails at 5", vt.false_alarm_status(3, 517, 0.02) == "PASS" and vt.false_alarm_status(5, 517, 0.02) == "FAIL")
check("the smallest n for a zero-hit 5% claim is 73", vt.wilson(0, 73)[1] <= 0.0501 and vt.wilson(0, 72)[1] > 0.0501)
check("a defect with no built detector is NOT MEASURED, never a failure", vt.recall_status("future_date", 0, 100, 0.95) == "NOT_MEASURED")
saved = vt.BUILT_DETECTORS
vt.BUILT_DETECTORS = ("future_date", "percent_scale")
check("a built type below the minimum n is NOT GATED", vt.recall_status("percent_scale", 20, 20, 0.85) == "NOT_GATED")
check("a built type at n >= minimum is judged on its point estimate", vt.recall_status("future_date", 96, 100, 0.95) == "PASS" and vt.recall_status("future_date", 94, 100, 0.95) == "FAIL")
vt.BUILT_DETECTORS = saved
check("nothing is built yet (so the first scorer run reports only what exists)", vt.BUILT_DETECTORS == ())

# ── definitions that must stay coherent ──
import clean_schema as csch
known = {c.value for c in csch.ReasonCode}
mapped = {r for rs in vt.ACCEPTED_REASONS.values() for r in rs}
check("every accepted reason is an existing ReasonCode or a declared NEW one", mapped <= known | set(vt.NEW_REASON_CODES), sorted(mapped - known - set(vt.NEW_REASON_CODES)))
check("the NEW reason codes are not in clean_schema yet (so adding them later cannot collide)", not (set(vt.NEW_REASON_CODES) & known), sorted(set(vt.NEW_REASON_CODES) & known))
check("every measured type has an accepted-reason mapping", set(vt.TARGETS["min_recall"]) <= set(vt.ACCEPTED_REASONS))
check("impossible_value is scoped to numeric rules (no date rule in the set)", all("date" not in r.lower() for r in vt.IMPOSSIBLE_RULES) and len(vt.IMPOSSIBLE_RULES) == 2)
check("structure and cross-column rules are outside the per-cell precedence", not ({"structure", "end_before_start", "orphan_fk", "swapped_values"} & set(vt.PRECEDENCE)))
check("the minimum sizes for new decoy columns clear the zero-hit 5% bar (>= 73)", all(vt.NEW_BENCH_MIN_LABELS[k] >= 73 for k in ("legit_format_mix", "free_text_column", "padded_fixed_width")))
check("the frozen-split policy is written down", "reported as a failure" in vt.FROZEN_POLICY and "no second split" in vt.FROZEN_POLICY)
check("fixed-width padding is an ask at 90% consistent padding", vt.TRIM_CONSISTENT_PADDING_ASK_AT == 0.90)

print()
if FAIL:
    print(f"{len(FAIL)} FAILED")
    sys.exit(1)
print("DONE — all checks passed.")
