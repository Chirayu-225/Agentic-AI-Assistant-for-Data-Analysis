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
  decoy   padded_fixed_width   fixed-width codes where the padding is meaningful (trimming would damage them)
  Sizes (per split, counted in COLUMNS): see NEW_BENCH_MIN_LABELS. They come from the false-alarm gate: a zero-hit result
  clears a 5% bar only from n = 73, so a decoy type with fewer columns could never be gated.
  Constraint: the generator change must leave every existing case unchanged. `python clean_bench.py verify` and the
  experiment's data hash in PREREG.json must still match. The frozen split has never been scored, so extending it
  is allowed now and only now.

HOW A BAR IS JUDGED (added after review, before any scorer run)
  RECALL unit   a defect is LOCATED when its cell lies inside a card whose reason code is one of ACCEPTED_REASONS for that
                type. That is the gated number. Cell-only recall (any card covers the cell) is reported beside it.
  RECALL gate   the point estimate must meet the bar AND n >= min_n_recall. Below that n the type is NOT GATED. A Wilson
                95% interval is printed beside every rate and the write-up may claim only what the interval supports.
                (A Wilson LOWER-bound gate was rejected: at the real dev sizes it demands near-perfect scores, for
                example 32 of 32 for percent_scale to clear 0.85, which would turn the bar into "perfect".)
  FALSE ALARMS  gated on the Wilson 95% UPPER bound, because "no more than x%" is a claim about a limit and a zero count
                on a small n proves little. A type whose upper bound exceeds its bar even at zero hits is NOT GATED.
                The unit for column-level decoys is the COLUMN, so one mistake on a column counts once.
  NOT MEASURED  a defect type whose detector is not listed in BUILT_DETECTORS is reported "not measured", never "fail".
  RANKING       cards are ranked by RANKING_KEY. The benchmark's labels and questions never take part in ranking.
  FROZEN        scored once, with the final knobs, aggregate only. A failure is reported as a failure. There is no
                retune-and-rescore, and no second frozen split to try again on.

HONEST LIMITS (to be shown beside every result)
  - Wrong values that fit the column's normal shape (41 typed as 14) pass every check. No design closes this.
  - The benchmark measures only the defect types it injects. Tables are synthetic, CSV only, and the same author
    wrote the defects and the detectors.
  - The expected bucket for each defect type is defined by the same author, so the routing bar is a check on consistency,
    not independent validation.
  - Bars that depend on injection difficulty apply to what the benchmark injects: outliers are x100 on money columns;
    impossible values are R1/R2 below. Milder outliers and other rules are not claimed.
  - unit_mismatch (0.80) is expected to be the first miss. It is not to be lowered after the result.
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
    "min_n_recall": 30,                        # below this many labels a type is NOT GATED
    "recall_unit": "cell inside a card with an accepted reason code (cell-only recall reported beside it)",
    "false_alarm_gate": "wilson_upper_95",
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
    "trim_whitespace is never auto-applied to a column whose padding is consistent (fixed-width codes): that is an ask",
)

AUTO_APPLY_ALLOWLIST = ("trim_whitespace", "add_normalized_copy")
# trim_whitespace is auto-applied only where padding is INCONSISTENT (a minority of values). If 90% or more of a
# column's values share identical padding it is formatting (fixed-width codes): that is an ask, never an auto-fix.
TRIM_CONSISTENT_PADDING_ASK_AT = 0.90

# Per-cell value detectors only. First match wins the routing; losers are kept as also_matched evidence on the winning
# card, and recall per type counts "any detector fired on this cell". Structure (reader findings) and cross-column
# rules (end<start, orphan keys, swapped columns) sit OUTSIDE this order: they are confirmed proposals, not cell-level.
PRECEDENCE = ("impossible_value", "future_date", "percent_scale", "unit_mismatch",
              "lost_leading_zero", "outlier_magnitude", "format_drift")

# Numeric-domain rules only. A future date is a date rule, so it is not in this set (or its recall would read zero).
IMPOSSIBLE_RULES = (
    "R1  a negative value in a count / quantity / stock column (a negative AMOUNT is a refund and is not flagged)",
    "R2  a value outside the bounded integer scale inferred from the column's bulk (e.g. a rating column whose bulk is 1-5)",
)

# ranking of cards on the review screen
RANKING_KEY = ("severity descending", "cell_count descending", "column position ascending")

# reason codes a located defect may carry. Existing codes live in clean_schema.ReasonCode; NEW ones are added there in
# the commit of the first detector that needs them.
ACCEPTED_REASONS = {
    "impossible_value": ("IMPLAUSIBLE_RANGE",),
    "future_date": ("FUTURE_DATE",),
    "percent_scale": ("PERCENT_SCALE",),
    "unit_mismatch": ("UNIT_MISMATCH",),
    "lost_leading_zero": ("ZERO_PADDING_LOST",),
    "outlier_magnitude": ("OUTLIER_ROBUST",),
    "format_drift": ("FORMAT_DRIFT",),
}
NEW_REASON_CODES = ("FUTURE_DATE", "PERCENT_SCALE", "ZERO_PADDING_LOST", "FORMAT_DRIFT", "FORMAT_VARIANCE")

# defect types whose detector exists and may be scored. Updated in the same commit as each detector. Empty today.
BUILT_DETECTORS: tuple = ()

# minimum labels per split for the new benchmark pieces (columns for column-level decoys)
NEW_BENCH_MIN_LABELS = {"legit_format_mix": 80, "free_text_column": 80, "padded_fixed_width": 80, "format_drift": 40}

# ── judging helpers: pure arithmetic ─────────────────────────────────────────────────────────────
import math

_Z95 = 1.959964


def wilson(k: int, n: int) -> tuple:
    """95% Wilson score interval for k successes in n trials -> (lower, upper)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + _Z95 ** 2 / n
    centre = (p + _Z95 ** 2 / (2 * n)) / d
    half = _Z95 * math.sqrt(p * (1 - p) / n + _Z95 ** 2 / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def recall_status(defect: str, hits: int, n: int, bar: float) -> str:
    """PASS / FAIL / NOT_GATED / NOT_MEASURED for one defect type."""
    if defect not in BUILT_DETECTORS:
        return "NOT_MEASURED"
    if n < TARGETS["min_n_recall"]:
        return "NOT_GATED"
    return "PASS" if hits / n >= bar else "FAIL"


def false_alarm_status(hits: int, n: int, bar: float) -> str:
    """PASS / FAIL / NOT_GATED on the Wilson upper bound. NOT_GATED when even zero hits could not clear the bar."""
    if wilson(0, n)[1] > bar:
        return "NOT_GATED"
    return "PASS" if wilson(hits, n)[1] <= bar else "FAIL"


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

FROZEN_POLICY = "scored once with the final knobs; a failure is reported as a failure; no retune-and-rescore; no second split"

# The final project review. Friday, 13 November 2026. Set by Farrel. A miss is a prompt to cut scope, not to move the date.
# What is demoed on this date is a SMALLER slice than the bars above describe; nothing below is claimed as measured
# until the benchmark scorer has actually been run on it.
SOFT_DEADLINE = "2026-11-13"
