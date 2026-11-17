"""
defect_prompts.py - what the AI cleaning step is shown and asked.

The cleaning agent (cleaning_agent.py) first diagnoses defects, then asks for a fix:

build_defect_profile()          -> per-column facts about suspect values (wider than the
                                   analysis profile: it is not capped at six columns)
format_defect_profile_for_llm() -> that profile as text for the model
build_defect_diagnosis_prompt() -> step 1: list the defects
build_defect_fix_prompt()       -> step 2: write the pandas code that fixes them

Moved here from analysis/profiler.py so the cleaning package does not depend on the
analysis package (see tests/architecture/test_package_boundaries.py).
"""

import numpy as np
import pandas as pd


def build_defect_profile(dataframe: pd.DataFrame, low_cardinality_max: int = 50) -> dict:
    """
    Wider than build_dataset_profile() above — that one caps at 6
    categorical columns and top-5 values per column, because it was
    built for the insight-generation feature's cost budget, not for
    defect detection. A defect-detection agent needs to see EVERY
    column (a spelling inconsistency in column 15 is just as real as
    one in column 2) and every distinct value where that's cheap — a
    rare typo is often exactly the value that won't make a top-5 list
    by frequency, which is the whole reason it survived undetected.

    Numeric stats / outliers / correlations / nulls are unchanged from
    build_dataset_profile — those were never capped, only the
    categorical section was.

    Cost is still bounded deliberately: a column with more than
    low_cardinality_max distinct values is very likely free text or an
    identifier, not a bounded category — enumerating "all 3,000 messy
    job titles" would blow the token budget for no real benefit, so
    those columns get a top-10 sample and a note instead of the full set.
    """
    sample_df = (
        dataframe.sample(min(len(dataframe), 5000), random_state=42)
        if len(dataframe) > 5000 else dataframe
    )

    profile: dict = {"rows": dataframe.shape[0], "columns": dataframe.shape[1]}

    null_counts = dataframe.isnull().sum()
    profile["nulls"] = {col: int(n) for col, n in null_counts.items() if n > 0}

    num_df = sample_df.select_dtypes(include="number")
    if not num_df.empty:
        profile["numeric_stats"] = num_df.describe().round(2).to_dict()

    outliers: dict = {}
    for col in num_df.columns:
        q1, q3 = sample_df[col].quantile(0.25), sample_df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            n_out = sample_df[(sample_df[col] < q1 - 1.5 * iqr) | (sample_df[col] > q3 + 1.5 * iqr)].shape[0]
            if n_out > 0:
                outliers[col] = n_out
    profile["outliers"] = outliers

    if len(num_df.columns) >= 2:
        corr = num_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        strong = upper.stack().reset_index().rename(columns={"level_0": "col_a", "level_1": "col_b", 0: "corr"}).query("corr > 0.5")
        profile["strong_correlations"] = [
            {"col_a": r.col_a, "col_b": r.col_b, "corr": round(r.corr, 3)} for r in strong.itertuples()
        ]

    # No 6-column cap, no top-5-only cap — the point of this function.
    categorical: dict = {}
    for col in sample_df.select_dtypes(include=["object", "category"]).columns:
        counts = sample_df[col].value_counts()
        if len(counts) <= low_cardinality_max:
            categorical[col] = {
                "all_distinct_values": counts.to_dict(),
                "note": None,
            }
        else:
            categorical[col] = {
                "all_distinct_values": counts.head(10).to_dict(),
                "note": f"{len(counts) - 10} more distinct values not shown — "
                        f"high cardinality, likely free text or an identifier",
            }
    profile["categorical_values"] = categorical

    return profile


def format_defect_profile_for_llm(profile: dict) -> str:
    from security.pii_redaction import withhold_personal_in_profile
    profile = withhold_personal_in_profile(profile)
    lines = [f"Dataset: {profile['rows']:,} rows × {profile['columns']} columns", ""]

    if profile["nulls"]:
        lines.append("Missing values by column:")
        for col, n in profile["nulls"].items():
            lines.append(f"  {col}: {n} missing ({round(n / profile['rows'] * 100, 1)}%)")
        lines.append("")

    if profile.get("numeric_stats"):
        lines.append("Numeric column statistics:")
        for col, stats in profile["numeric_stats"].items():
            lines.append(f"  {col}: min={stats.get('min')} max={stats.get('max')} "
                         f"mean={stats.get('mean')} std={stats.get('std')}")
        lines.append("")

    if profile.get("outliers"):
        lines.append("Statistical outliers (IQR method):")
        for col, n in profile["outliers"].items():
            lines.append(f"  {col}: {n} outlier row(s)")
        lines.append("")

    if profile.get("strong_correlations"):
        lines.append("Strong numeric correlations:")
        for c in profile["strong_correlations"]:
            lines.append(f"  {c['col_a']} <-> {c['col_b']}: {c['corr']}")
        lines.append("")

    if profile.get("categorical_values"):
        lines.append("Categorical column values (ALL distinct values shown for low-cardinality columns):")
        for col, info in profile["categorical_values"].items():
            values_str = ", ".join(f"'{v}' (x{c})" for v, c in list(info["all_distinct_values"].items())[:30])
            note = f" [{info['note']}]" if info["note"] else ""
            lines.append(f"  {col}: {values_str}{note}")
        lines.append("")

    text = "\n".join(lines)

    # AI SECURITY: this is the WIDEST data-exposure surface in the app —
    # unlike format_profile_for_llm's 6-column/top-5 cap, this function
    # deliberately shows EVERY column and EVERY distinct value for
    # low-cardinality columns (that's its whole point, see
    # build_defect_profile's docstring), so a customer-list or HR-export
    # style dataset would send real emails/phone numbers/IDs straight to
    # the LLM without this. Same redact-then-fence pattern as
    # format_profile_for_llm above — see pii_redaction.py and
    # prompt_injection_guard.py's module docstrings.
    from security.pii_redaction import redact_pii_logged
    text = redact_pii_logged(text, context="defect profile (cleaning agent)")

    from security.prompt_injection_guard import wrap_untrusted
    return wrap_untrusted(text, source_label="defect profile")


def build_defect_diagnosis_prompt(profile_str: str) -> str:
    """First step of the data quality agent: find what's actually wrong,
    before deciding what to do about it. Deliberately looks for the
    class of defect a purely deterministic/regex pass structurally
    can't catch — cleaning.py already handles currency symbols,
    percentages, thousands separators, and identifier columns; this
    prompt exists for everything else."""
    return f"""You are a senior data quality analyst auditing a dataset
before it's used for analysis. A deterministic rule-based pass has
already run and fixed currency symbols, percentages, thousands
separators, and identifier-column mis-typing — do not re-report those.

Look specifically for what pattern-matching CAN'T catch:
- Inconsistent spellings/casing of what should be the same category
  (e.g. "USA" / "U.S." / "United States" all meaning one country).
- Implausible values that aren't malformed, just wrong (a 1-5 rating
  scale with some values of 500; a percentage over 100).
- Values that contradict what the column and its neighbours suggest.
- Genuinely meaningful missing values (not just "fill with median") —
  where the ABSENCE itself likely means something specific.

DATASET PROFILE:
{profile_str}

List up to 8 real, concrete defects you actually find in the data
above — do not invent issues to fill the list; if there are genuinely
fewer than 8, list fewer, and if you find none, say exactly "NO DEFECTS
FOUND" and nothing else.

Otherwise respond ONLY with defect blocks in this exact plain-text
format — no markdown bold, no numbering, no commentary before or after:

COLUMN: <exact column name, or "cross-column" if it spans several>
TYPE: <one of: missing_values, inconsistent_categories, outlier, mixed_format, structural>
EVIDENCE: <the specific observation from the profile above proving this is real>
ACTION: <one sentence: what should be done about it>

(repeat one block per defect, separated by a blank line)
"""


def build_defect_fix_prompt(defects: list[dict], schema_text: str) -> str:
    """Second step: given the diagnosed defects, decide and write the
    actual fix. Kept as a SEPARATE call from diagnosis deliberately —
    the diagnosis becomes an inspectable, reportable artifact in its
    own right (what was found, with evidence), independent of whether
    the fix step later succeeds or fails."""
    lines = [
        "You are fixing the following data quality defects on `df` (a "
        "pandas DataFrame that's already been through deterministic "
        "rule-based cleaning). The user's data is the source of truth: "
        "you may tidy how a value is WRITTEN, but you may never change "
        "WHAT it says. Your code is checked afterwards and any column "
        "that breaks these rules is put back exactly as it was.",
        "",
        "HARD RULES:",
        "- NEVER replace an existing value with NaN or None.",
        "- NEVER fill missing values (no 'Unknown', no 0, no mean/median, "
        "no ffill/bfill/interpolate).",
        "- NEVER rename or merge categories (do not turn 'Planes' into "
        "'Aircraft', and do not map several spellings onto one).",
        "- NEVER drop rows or columns.",
        "- NEVER reorder or rename rows or columns, and never change the "
        "index.",
        "- ONLY add new boolean flag columns (for example sales_is_outlier), "
        "strip whitespace, normalize case in low-cardinality text columns, "
        "or parse text to numbers or dates when EVERY value parses.",
        "- A flag column must be boolean with False (never empty/None/NaN) "
        "for every row that is not flagged, e.g. df['x'] > 100 rather than "
        "a column that is only filled in for the flagged rows.",
        "- Anything else — merging spelling variants, imputing, correcting "
        "an outlier, deciding what a blank means — is a human's decision. "
        "Leave that data unchanged and, on that defect's REASONING line, "
        "write: proposal only: <what a human should decide>.",
        "",
        f"Columns available: {schema_text}",
        "",
        "Defects to fix:",
    ]
    for i, d in enumerate(defects, 1):
        lines.append(f"{i}. [{d['type']}] '{d['column']}': {d['evidence']} → {d['action']}")
    lines.append("")
    lines.append(
        "Respond in exactly this format — a reasoning block, then ONE "
        "python code block implementing every fix, assigning the result "
        "to `cleaned_df`:\n\n"
        "REASONING:\n"
        "- <defect>: <what you did> — <why>\n"
        "- <defect>: proposal only: <what a human should decide>\n"
        "(one line per defect, nothing else in this section; a "
        "'proposal only' line is shown to the user and never run)\n\n"
        "```python\n"
        "cleaned_df = df.copy()\n"
        "# ... your fixes here, one defect at a time ...\n"
        "```\n\n"
        "Rules: only use pandas/numpy operations already available as "
        "df/pd/np. No imports. No print(). Only touch columns relevant "
        "to the defects listed above. Keep the code SHORT (under 40 lines) "
        "and never write one mapping entry per distinct value. If a column is "
        "mostly empty, leave it as-is and note that in REASONING. "
        "Iterating a DataFrame (`for c in df.columns`) gives column NAMES "
        "(plain strings), so use df[c] to get the column; select text columns "
        "with df.select_dtypes(include=['object','string'])."
    )
    return "\n".join(lines)
