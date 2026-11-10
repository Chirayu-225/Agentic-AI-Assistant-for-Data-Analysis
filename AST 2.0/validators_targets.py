"""
validators_targets.py — exit criteria for SLICE 1 of the validation pipeline, written BEFORE any detector
or router code exists and before anything has been scored. This file is the contract the code is judged by.

WHAT SLICE 1 IS
  Profile -> Detect -> Triage -> (one batched LLM step, a later slice). Slice 1 is the deterministic half:
    1. a generic SHAPE detector that works on any text / id / code column, not only phone numbers
    2. single-column value checks: impossible values, robust outliers, future dates, lost leading zeros,
       percent scale, unit mismatch
    3. the ROUTER, which puts every finding into exactly one bucket (below)
  Not in slice 1: cross-column rules (end<start, amount=qty*price, swapped columns, orphan keys -- these are
  proposals the user confirms, never blind detections), the LLM step, the review screen, drift across uploads, .xlsx.

THE FOUR BUCKETS (decided by code, never by the model)
  clean        fits the column's inferred contract and no detector fired                    -> nobody
  explained    a detector fired and has ONE clear default (e.g. blank a future date)         -> proposal, approved in bulk
  ambiguous    a detector fired but several fixes are plausible                              -> the agent proposes, the user decides
  unexplained  the cell breaks the column's inferred contract and no detector says why       -> the agent proposes, the user decides
  Findings are grouped into CARDS by (column, reason, shape). The agent only ever sees cards, never single cells.
  FORMAT VARIANCE (two or more legitimate styles) is NOT an invalid value: it gets a note and an offer of a
  normalized COPY column, and no per-row flags.

BARS vs KNOBS (the rule that keeps this honest)
  BARS   (TARGETS below) are pass/fail lines, fixed now. They are my priors, not numbers derived from data. They may
         be changed only BEFORE the first scorer run, by editing this file in a commit of its own. After the first
         run a missed bar is information about the detector, not a number to adjust. test_validators_targets.py hashes
         the bars, so an edit shows up as a failing test and a visible diff (a tripwire, not a lock).
  KNOBS  (SHAPE_KNOBS) are detector parameters, such as how dominant a shape must be to count as drift. They cannot be
         known without looking at data, so they MAY be tuned on the DEV split only, and every change must be appended
         to KNOB_LOG with the reason and the dev numbers before and after. The frozen split is scored once, with the
         final knobs, aggregate only, through the existing guard (clean_score.py). Tuning on dev cannot memorise the
         frozen vocabulary: the two splits use disjoint formats, sentinels and sizes.

BENCHMARK ADDITIONS NEEDED BEFORE ANY DETECTOR (new profile "validators"; existing profiles stay byte-identical)
  defect  format_drift        a column with one dominant shape and a 3-10% minority in another (ids, codes, postcodes, phone-like)
  decoy   legit_format_mix    two or more legitimate styles in one column (e.g. international phone numbers)
  decoy   free_text_column    names / notes whose shapes vary naturally (must never be shape-flagged)
  Constraint: the generator change must leave every existing case unchanged. `python clean_bench.py verify` and the
  experiment's data hash in PREREG.json must still match. The frozen split has never been scored, so extending it
  is allowed now and only now.

HONEST LIMITS (to be shown beside every result)
  - Wrong values that fit the column's normal shape (41 typed as 14) pass every check. No design closes this.
  - The benchmark measures only the defect types it injects. Tables are synthetic, CSV only, and the same author
    wrote the defects and the detectors.
  - Slice 1 measures detection and routing. It does not measure whether proposed fixes are right, or whether the
    cleaned answers improve. That is the experiment's question, and it is a separate claim.
"""

# ── the bars ────────────────────────────────────────────────────────────────────────────────────
TARGETS = {
    # located = flagged at the right source coordinates with a reason in the right class, as clean_score judges it.
    # Reported per defect type with its n. Never pooled.
    "min_recall": {
        "future_date": 0.95,
        "impossible_value": 0.90,
        "lost_leading_zero": 0.85,
        "percent_scale": 0.85,
        "format_drift": 0.85,
        "unit_mismatch": 0.80,
        "outlier_magnitude": 0.80,
    },
    # false alarms. Denominator: decoy labels (a flag, change or ask on a decoy cell is a hit).
    "max_decoy_hit_rate": 0.02,
    "max_decoy_hit_rate_per_type": 0.05,       # small n per type: always report n beside the rate
    # the 0%-defect control files carry decoys but no defects
    "max_cards_per_control_mean": 2.0,
    "max_cards_any_control": 4,
    "max_control_cells_flagged": 0.005,        # share of all cells in control files
    # alert fatigue, over the dirty dev cases
    "max_cards_median_dirty": 12,
    "max_cards_any_dirty": 25,
    "min_top10_coverage": 0.70,                # share of located defect cells that sit inside the first 10 ranked cards
    # damage: auto-applied changes may only come from the allowlist and must never destroy a correct cell
    "max_auto_cells_corrupted": 0,
    # segregation quality: located defect cells that land in the bucket the benchmark tags as expected
    "min_routed_as_expected": 0.90,
    # the LLM step
    "max_llm_calls_per_table": 1,
    "max_llm_calls_when_nothing_ambiguous": 0,
    # speed, for the detect stage alone (priors, to be checked on the real host)
    "max_detect_seconds_600_rows": 1.0,
    "max_detect_seconds_100k_rows": 10.0,
}

# ── router invariants: not statistics, each is a test that must hold on every input ───────────────
ROUTER_INVARIANTS = (
    "every finding lands in exactly one of explained / ambiguous / unexplained",
    "auto-apply comes only from AUTO_APPLY_ALLOWLIST and never overwrites a source value",
    "zero LLM calls when no card is ambiguous or unexplained; at most one call per table otherwise",
    "the LLM packet is built only from LlmVisible fields; planted canary strings (emails, phone numbers) never appear in it",
    "a cell flagged by two detectors receives exactly one proposal, chosen by PRECEDENCE",
    "same input gives byte-identical findings in the same order",
    "a table under min_rows_for_statistics rows gets no statistical flag (outlier, shape dominance)",
)

AUTO_APPLY_ALLOWLIST = ("trim_whitespace", "add_normalized_copy")

# first match wins when one cell is flagged by several detectors
PRECEDENCE = ("structure", "impossible_value", "future_date", "percent_scale", "unit_mismatch",
              "lost_leading_zero", "outlier_magnitude", "format_drift")

# ── knobs: starting values, tunable on DEV only, every change logged below ─────────────────────────
SHAPE_KNOBS = {
    "signature": "digit->9, upper->A, lower->a, other characters kept",
    "free_text_distinct_shapes_abs": 20,       # skip a column when distinct shapes exceed max(abs, frac * non-null)
    "free_text_distinct_shapes_frac": 0.05,
    "drift_dominant_min": 0.90,                # one shape at or above this share -> the minority is DRIFT (flag, ask)
    "variance_second_shape_min": 0.10,         # two shapes each at or above this share -> VARIANCE (note, copy column)
    "min_rows_for_statistics": 100,
}
# The thresholds of the other detectors (outlier robust-z, unit ratios, percent split) are fixed in the same commit
# that adds the detector, BEFORE its first dev run, and are added to this file then.

KNOB_LOG: list = []   # each entry: {"knob", "from", "to", "reason", "dev_before", "dev_after", "date"}

# The date by which a demoable slice is due. None until Farrel sets it.
SOFT_DEADLINE = None
