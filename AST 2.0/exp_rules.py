"""
exp_rules.py — PRE-REGISTRATION for the experiment "does cleaning change analysis answers?".
Fixed before any LLM run; the harness refuses to run an LLM pass if these, the question set or the data change
(see PREREG.json). Fix harness bugs after a pass if you must; do NOT edit questions or thresholds once you have
seen results.

QUESTION
    Does cleaning the file change the answers a user gets, and how much of the change comes from Stage 0?

ARMS (same questions, same model, same temperature; all arms of a question run together, same day)
    clean   the ATTAINABLE table (ceiling): what a perfect cleaner could deliver from the dirty file. Every recoverable
            defect is repaired; cells whose original value was destroyed (sentinels, future dates, negative quantities,
            end-before-start dates, orphan keys) are blanked, because no cleaner, reviewer or model can recover them.
            `expected` is computed on this table, so headroom measures what cleaning can actually win, not information
            the file no longer contains. The full-truth answer is kept as `expected_true` and `info_lost` in the report.
    dirty   the injected file through today's reader + cleaning.py (what a user gets now)
    stage0  the new reader + the same cleaning.py, fed to the unchanged analysis path
    (a fourth arm, "Stage 0 + default proposals", does not exist yet: there are no validators to produce proposals)

TIERS
    single    40 questions, one defect type per table: a wrong answer has one cause
    compound  10 questions on tables with 2-3 stacked defects, like real files
    The single tier is the EASIEST case for a model that compensates for dirty data, so its headroom is a LOWER
    BOUND. "Deprioritize" may be concluded only if the compound tier agrees.

THE SHAKEDOWN IS NOT A VERDICT
    The first pass is 1 repeat of the single tier. Run-to-run noise on this stack is about 15%, so one repeat cannot
    support a decision. It exists to shake out harness bugs, to measure tokens per call, and to show whether the gap is
    large or near zero. No decision rule is evaluated on it, and it must not be quoted as the result.

SCORING (deterministic, no LLM judge): eval_harness.check_answer. Counts and years must be exact. Sums, averages,
maxima and shares must be within 0.1% (relative). Average service in years within 1%.

DECISION RULES (apply to the full run only: 3 repeats, single + compound tiers)
  Per question and arm, the repeats collapse to a majority (strictly more than half of the non-infra repeats correct;
  a tie counts as wrong). Infra errors are excluded and counted, never substituted. A crash (the file could not be
  read) is a wrong answer, not an infra error.

  1. HEADROOM. Condition on questions the CLEAN arm answers correctly, so model error does not leak into the cleaning
     signal. Over the value-level single-tier questions pooled together, headroom = share of those questions the
     dirty arm gets wrong. Interval: paired bootstrap over questions, 5000 resamples, seed 20261105, 95%.
       - headroom < 10 points and the interval includes zero  -> DEPRIORITIZE validators and the review screen,
         and reposition cleaning as transparency. Only if the compound tier agrees (its pooled headroom is also
         under 10 with an interval that includes zero). If the compound tier disagrees: inconclusive, add questions.
       - headroom >= 10 points and the interval excludes zero -> BUILD the validators, ranked by per-type headroom.
       - anything else                                         -> ADD repeats or questions before deciding.
     Per-type numbers (about 2 questions each) are for ranking and reporting only.

  2. STAGE 0 WIRING (stage0 vs dirty). Wire it in only if ALL hold:
       - on the structural questions it matches or beats dirty on every defect type;
       - zero new crashes (no question the dirty arm could read that stage0 cannot);
       - on all other questions it is no more than ONE question net worse than dirty.
     The same condition is the retirement criterion for the old reader.

  3. GAP CLOSED = (stage0 - dirty) / (clean - dirty), reported only where clean - dirty >= 20 points.

SILENT-WRONG: a wrong, executed answer with no flag or exclusion note. run_analysis currently discards the cleaning
log, so no response carries a note and silent-wrong equals "wrong and executed". The metric starts to discriminate
once flags reach the answer.

CAVEATS TO STATE WITH ANY RESULT
    - the same hand wrote the questions, the injectors and the detectors: this measures KNOWN defect classes only
    - CSV only: no .xlsx evidence, and most real uploads are probably .xlsx
    - synthetic tables of 600 rows, three table families, plain locale
    - one model (pinned in PREREG.json), one provider
    - the ceiling assumes unrecoverable cells are blanked, not guessed (a defensible choice, but a choice)
    - an identifier column loses its leading zeros when the analysis path re-reads the Stage 0 table
"""

SEED = 20261105
BOOTSTRAP = 5000

RULES = {
    "version": 1,
    "arms": ["clean", "dirty", "stage0"],
    "structural_tags": ["footer", "total", "header", "ragged", "bomtrail"],
    "headroom": {"min_points": 10.0, "ci": 0.95, "bootstrap": BOOTSTRAP, "seed": SEED,
                 "condition": "clean-arm majority answer correct", "pooled_over": "value-level single-tier questions"},
    "stage0_wiring": {"structural": "matches or beats dirty on every defect type", "new_crashes": 0,
                      "non_structural_net_worse_max": 1},
    "gap_closed_min_clean_minus_dirty": 20.0,
    "majority": "strictly more than half of non-infra repeats correct; tie = wrong",
    "tolerances": {"count": 0.0, "year": 0.0, "sum_avg_max_share": 0.001, "tenure": 0.01},
    "full_run": {"repeats": 3, "tiers": ["single", "compound"]},
    "shakedown": {"repeats": 1, "tiers": ["single"], "decision_rules_apply": False},
}
