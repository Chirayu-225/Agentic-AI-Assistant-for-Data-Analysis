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
EXPECTED_BARS_SHA = "fea2a46b1f004f26"
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
check("the invariants are listed (7)", len(vt.ROUTER_INVARIANTS) == 7)

print()
if FAIL:
    print(f"{len(FAIL)} FAILED")
    sys.exit(1)
print("DONE — all checks passed.")
