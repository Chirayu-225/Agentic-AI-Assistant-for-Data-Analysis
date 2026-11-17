"""
stage0_targets.py — exit criteria for the Stage 0 reader, written BEFORE the reader existed and
before its first scorer run. Thresholds are fixed here; they are not to be edited after seeing a
result. A target that is missed is information about which heuristic to revisit, not a number to
adjust.

Source: the dev baseline of the existing reader + rule_based_clean (see clean_score.py --adapter legacy).
Dev split only. The frozen split gets its first look when the full pipeline exists.

    python clean_score.py --adapter stage0     # prints PASS / MISS against these targets

DEFINITION OF "stage0" in the scorer: the new reader -> an all-text table -> the SAME rule_based_clean
the baseline used, so the comparison isolates what the reader changes.

TARGETS
  crashes         0 of 15 dev cases fail to produce a table                       (baseline: 5 crashed)
  controls        the reader alone is the identity on every clean control: same rows, zero flags,
                  zero asks, zero rows removed
  asks            the reader asks about at most 2 of the 15 dev cases, and about NONE of the controls
                  (over-asking is the failure mode of this design)
  structure       per defect type, correct as scored (crashed cases count as misses):
                      header_offset 100%   bom 100%   trailing_delimiter 100%
                      footer_note  >= 95%  total_row 100%  blank_row >= 98%  ragged_row >= 90%
  damage          cells corrupted <= 0.1% of originally-correct cells            (baseline: 1.2%)
                  clean/decoy rows wrongly removed: 0                             (baseline: 0)
                  decoys touched: 0
                  control files: cells touched <= 0.5%                            (baseline: 1.1%)
  no regression   every type below must stay within 2 points of its baseline rate on READABLE files
                  (baseline readable-file rate in brackets), now measured over ALL cases:
                      sentinel_null [100%]  number_as_text [100%]  category_variant [100%]
                      compound [100%]  dup_exact [100%]  number_formatted_text [100%]
                      whitespace [85.2%]
  out of scope    these are expected to stay near their baseline (they need validators, not a reader):
                  swapped_values, outlier_magnitude, impossible_value, future_date, orphan_fk,
                  end_before_start, unit_mismatch, percent_scale, date_mixed_format, lost_leading_zero
                  (a cell that already lost its zeros in the file needs a fix proposal, which is Stage 1)
"""

TARGETS = {
    "max_crashed_cases": 0,
    "max_asked_cases": 2,
    "max_asked_controls": 0,
    "max_cells_corrupted_rate": 0.001,
    "max_clean_rows_removed": 0,
    "max_decoy_hits": 0,
    "max_control_cells_touched": 0.005,
    "min_correct": {"header_offset": 1.0, "bom": 1.0, "trailing_delimiter": 1.0, "footer_note": 0.95,
                    "total_row": 1.0, "blank_row": 0.98, "ragged_row": 0.90},
    "no_regression": {"sentinel_null": 1.0, "number_as_text": 1.0, "category_variant": 1.0, "compound": 1.0,
                      "dup_exact": 1.0, "number_formatted_text": 1.0, "whitespace": 0.852},
    "regression_tolerance": 0.02,
}


def check(result: dict) -> list:
    """-> [(name, passed, detail)] against TARGETS, for a scorer result."""
    t, d, out = TARGETS, result["by_defect"], []
    out.append(("no crashed cases", result["cases_failed"] <= t["max_crashed_cases"], f"{result['cases_failed']} crashed"))
    g = result.get("gate") or {}
    out.append((f"asks on at most {t['max_asked_cases']} cases", g.get("asked", 0) <= t["max_asked_cases"], f"asked on {g.get('asked', 0)}"))
    out.append(("asks on no control", g.get("asked_controls", 0) <= t["max_asked_controls"], f"{g.get('asked_controls', 0)} controls asked"))
    for k, floor in t["min_correct"].items():
        v = d.get(k)
        rate = None if not v or not v["scored"] else v["correct"] / v["scored"]
        out.append((f"{k} >= {floor:.0%}", rate is not None and rate >= floor, "n/a" if rate is None else f"{rate:.1%} of {v['scored']}"))
    dm = result["damage"]
    cr = dm["cells_corrupted_rate"]
    out.append((f"cells corrupted <= {t['max_cells_corrupted_rate']:.1%}", cr is not None and cr <= t["max_cells_corrupted_rate"], f"{(cr or 0):.2%}"))
    wrong = dm["rows_wrongly_removed_reported"] + dm["rows_wrongly_removed_silent"]
    out.append(("no clean row removed", wrong <= t["max_clean_rows_removed"], f"{wrong} removed"))
    hits = sum(v["hits"] for v in result["by_decoy"].values())
    out.append(("no decoy touched", hits <= t["max_decoy_hits"], f"{hits} hits"))
    ct = result["controls"]["cells_touched_rate"]
    out.append((f"control cells touched <= {t['max_control_cells_touched']:.1%}", ct is not None and ct <= t["max_control_cells_touched"], f"{(ct or 0):.2%}"))
    for k, base in t["no_regression"].items():
        v = d.get(k)
        rate = None if not v or not v["scored"] else v["correct"] / v["scored"]
        out.append((f"{k} stays >= {base - t['regression_tolerance']:.1%}", rate is not None and rate >= base - t["regression_tolerance"],
                    "n/a" if rate is None else f"{rate:.1%}"))
    return out
