"""
profiler.py — Dataset profiling and insight prompt generation.

build_dataset_profile()     → structured dict (cached)
format_profile_for_llm()    → LLM-readable string
build_insight_prompt()      → context-aware prompt: result + full dataset
build_auto_analyze_prompt() → proactive dataset intelligence prompt
build_explain_prompt()      → drill-down explanation prompt
"""

import pandas as pd
import numpy as np
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PROFILER
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_dataset_profile(dataframe: pd.DataFrame, _cache_key: str = "") -> dict:
    """
    Compute a structured statistical profile. Cached by Streamlit.
    Pass _cache_key="cleaned" after cleaning to force a fresh profile
    and prevent stale raw-data stats being used for insights.
    Expensive ops use a 5000-row sample for large datasets.
    """
    profile: dict = {
        "rows":    dataframe.shape[0],
        "columns": dataframe.shape[1],
    }

    sample_df = (
        dataframe.sample(min(len(dataframe), 5000), random_state=42)
        if len(dataframe) > 5000 else dataframe
    )

    # Nulls — full dataset (cheap)
    null_counts = dataframe.isnull().sum()
    profile["nulls"] = {col: int(n) for col, n in null_counts.items() if n > 0}
    profile["null_pct"] = {
        col: round(n / len(dataframe) * 100, 1)
        for col, n in profile["nulls"].items()
    }

    # Numeric stats — sample
    num_df = sample_df.select_dtypes(include="number")
    if not num_df.empty:
        profile["numeric_stats"] = num_df.describe().round(2).to_dict()

    # Top categories — sample, cap at 6 cols
    cat_cols = sample_df.select_dtypes(include=["object", "category"]).columns
    profile["top_categories"] = {
        col: sample_df[col].value_counts().head(5).to_dict()
        for col in cat_cols[:6]
    }

    # Strong correlations — sample
    if len(num_df.columns) >= 2:
        corr  = num_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        strong = (
            upper.stack()
            .reset_index()
            .rename(columns={"level_0": "col_a", "level_1": "col_b", 0: "corr"})
            .query("corr > 0.5")
            .sort_values("corr", ascending=False)
            .head(5)
        )
        profile["strong_correlations"] = [
            {"col_a": r.col_a, "col_b": r.col_b, "corr": round(r.corr, 3)}
            for r in strong.itertuples()
        ]

    # Outliers via IQR — sample
    outliers: dict = {}
    for col in num_df.columns:
        q1  = sample_df[col].quantile(0.25)
        q3  = sample_df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            n_out = sample_df[
                (sample_df[col] < q1 - 1.5 * iqr) |
                (sample_df[col] > q3 + 1.5 * iqr)
            ].shape[0]
            if n_out > 0:
                outliers[col] = n_out
    profile["outliers"] = outliers

    return profile


def format_profile_for_llm(profile: dict) -> str:
    lines = [f"Dataset: {profile['rows']:,} rows × {profile['columns']} columns"]

    if profile.get("nulls"):
        null_str = ", ".join(
            f"{col} ({pct}%)" for col, pct in profile.get("null_pct", {}).items()
        )
        lines.append(f"Missing values: {null_str}")
    else:
        lines.append("Missing values: none")

    if profile.get("numeric_stats"):
        lines.append("\nNumeric column stats (mean / std / min / max):")
        for col, stats in profile["numeric_stats"].items():
            lines.append(
                f"  {col}: mean={stats.get('mean','?')}  "
                f"std={stats.get('std','?')}  "
                f"min={stats.get('min','?')}  "
                f"max={stats.get('max','?')}"
            )

    if profile.get("top_categories"):
        lines.append("\nTop category values:")
        for col, vals in profile["top_categories"].items():
            top = ", ".join(f"{k}({v})" for k, v in list(vals.items())[:3])
            lines.append(f"  {col}: {top}")

    if profile.get("strong_correlations"):
        lines.append("\nStrong correlations (>0.5):")
        for c in profile["strong_correlations"]:
            lines.append(f"  {c['col_a']} ↔ {c['col_b']}: {c['corr']}")

    if profile.get("outliers"):
        lines.append("\nOutliers detected (IQR method):")
        for col, n in profile["outliers"].items():
            lines.append(f"  {col}: {n} outlier(s)")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# INSIGHT PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

def build_insight_prompt(user_query: str, result_profile_str: str,
                         dataset_profile_str: str) -> str:
    """
    Context-aware: passes BOTH the result profile AND the full dataset
    profile so the model can compare the result against broader context.
    Strict actionability rules — no data quality complaints, no raw descriptions.
    """
    return f"""You are a senior business analyst presenting findings to a CEO.
The data has already been cleaned. Do NOT mention nulls, missing values, data types,
or data quality. The audience does not care about the data — they care about the business.

A user asked: "{user_query}"

ANALYSIS RESULT:
{result_profile_str}

FULL DATASET CONTEXT:
{dataset_profile_str}

Write 3 to 5 BUSINESS insights ranked by impact. Each insight must:
1. State a specific, quantified business finding (revenue, volume, rank, gap, trend)
2. Immediately follow it with a concrete business action

CORRECT examples:
1• Classic Cars generates 39% of total revenue but only 18% of order volume — increase unit price or upsell accessories to this segment
2• Motorcycles have the lowest average order value ($950) vs the dataset mean ($3,200) — investigate pricing strategy or target higher-value customers
3• Q4 sales are 2.3× higher than Q1 — align inventory procurement and staffing to seasonal demand pattern

WRONG examples (do NOT write these):
✗ The dataset contains 2,823 rows and 25 columns
✗ The sales column has 81 outliers detected using IQR
✗ There are missing values in addressline2
✗ The data shows that Classic Cars is the most common category

SELF-CHECK before writing each insight: ask yourself —
"Would a CEO find this useful for making a decision?" If no, rewrite it.
"Does it tell them what to DO, not just what IS?" If no, add the action.
"Does it mention the data structure rather than the business?" If yes, delete it.

Format: 1• [quantified business finding] — [specific action to take]
"""


def build_auto_analyze_prompt(dataset_profile_str: str) -> str:
    return f"""You are a senior business analyst presenting to a CEO for the first time.
The data has already been cleaned and standardised. Do NOT mention nulls, missing values,
data types, or anything about data quality. Focus entirely on business findings.

DATASET PROFILE:
{dataset_profile_str}

Write the 5 most impactful BUSINESS insights from this dataset, ranked by business value.

Each insight must answer: "What does this mean for the business, and what should we do?"

CORRECT examples:
1• Classic Cars(967 orders) outsells Motorcycles(331) by 3× — concentrate sales and marketing resources on this category
2• The top 2 product lines contribute over 65% of order volume — diversification risk is high; develop the bottom 3 lines
3• Sales std deviation ($1.2M) is 40% of the mean — highly inconsistent performance; identify what drives the top-quartile transactions

WRONG examples (never write these):
✗ The dataset has 2,823 rows
✗ The status column shows a skewed distribution
✗ There are missing values in the territory column
✗ The quantityordered column has 8 outliers

SELF-CHECK: Every insight must name a specific business implication and a specific action.
If an insight only describes the data without saying what to do about it — rewrite it.

Format: 1• [quantified business finding] — [specific action]
"""


def build_chart_plan_prompt(dataset_profile_str: str, insights_text: str) -> str:
    """Second step of auto-analyze: given the profile AND the insights just
    generated, pick the 3 visualisations that best make those insights
    land visually — not just "any 3 charts this data supports"."""
    return f"""You are a senior BI analyst choosing which visualisations to put
in front of a CEO. Data is already cleaned — do not propose a chart about
nulls, dtypes, or data quality.

DATASET PROFILE:
{dataset_profile_str}

TOP BUSINESS INSIGHTS ALREADY IDENTIFIED:
{insights_text}

Propose exactly 3 visualisations that best make the insights above visually
obvious at a glance.

Respond with ONLY 3 blocks in exactly this plain-text format — no markdown
bold/headers, no numbering, no commentary before or after, nothing else on
the page but these 3 blocks separated by a blank line:

TITLE: <short chart title, under 8 words>
SUPPORTS: <the insight number from above it visually supports>
SPEC: <one precise sentence: chart type, x, y, aggregation, sort order>

TITLE: <short chart title, under 8 words>
SUPPORTS: <the insight number from above it visually supports>
SPEC: <one precise sentence: chart type, x, y, aggregation, sort order>

TITLE: <short chart title, under 8 words>
SUPPORTS: <the insight number from above it visually supports>
SPEC: <one precise sentence: chart type, x, y, aggregation, sort order>

Rules:
- Every SPEC must be answerable using ONLY columns named in the dataset
  profile above — do not invent a column.
- Prefer the chart type that reveals the finding fastest: bar for
  comparison across categories, line for trend over time, scatter for a
  relationship between two numeric columns.
- Do not propose two charts that show essentially the same comparison.
"""


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

    return "\n".join(lines)


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
        "rule-based cleaning). For EACH defect, apply the most sensible "
        "fix given standard data-cleaning practice — e.g. for "
        "inconsistent categories, map every variant to one canonical "
        "spelling; for a meaningful missing value, use an honest label "
        "like 'Not Applicable' rather than forcing a numeric guess.",
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
        "(one line per defect, nothing else in this section)\n\n"
        "```python\n"
        "cleaned_df = df.copy()\n"
        "# ... your fixes here, one defect at a time ...\n"
        "```\n\n"
        "Rules: only use pandas/numpy operations already available as "
        "df/pd/np. No imports. No print(). Only touch columns relevant "
        "to the defects listed above."
    )
    return "\n".join(lines)


def build_followup_decision_prompt(schema_text: str, insights_text: str, prior_qa_text: str, question: str) -> str:
    """First step of a follow-up answer: decide whether answering this
    well needs real data computed directly from the dataset, or whether
    it's answerable from the insights already on screen. Kept as its
    own step (rather than letting the final-answer call freely invent
    numbers) because a computed, sandbox-executed value is real
    grounding — an LLM asserting a percentage in prose is not, even if
    it sounds confident.

    Explicitly allows UP TO 3 snippets, not just one — an early version
    of this capped it at a single fact, which meant a genuinely
    exploratory question ("explain why X happens") got exactly one shot
    at grounding before the answer step had to fall back to "I don't
    have enough detail." A real analyst facing a "why" question checks
    several angles (price, order size, regional concentration, timing)
    before concluding the data can't fully explain something — this
    should be able to do the same, while still never fabricating a
    number it didn't actually compute."""
    return f"""You are deciding HOW to answer a follow-up question about a
business analysis — not answering it yet.

Columns in the dataset: {schema_text}

Business insights already identified:
{insights_text}
{prior_qa_text}
User's follow-up question: {question}

If this is an explanatory "why" question, don't stop at one number —
think like an analyst would: what 2-3 different angles in the data
(e.g. price per unit, average order size, regional/customer
concentration, time trend) might help explain the pattern, even
partially? Write UP TO 3 short, SEPARATE python snippets, each in its
own ```python code block, each computing ONE specific angle. Each
snippet runs independently — assign each one's output to a variable
named `result` (not result_1/result_2 — each block gets its own fresh
run, so reusing the same name each time is correct, not a conflict).
It's fine if some angles turn out uninformative; computing them is
still more honest than guessing.

If the question is a straightforward business-reasoning question the
insights above already answer, or the dataset genuinely has no columns
that could inform any angle of it, respond with exactly this and
nothing else:
NO_CODE_NEEDED

Available libraries if you do write code (already imported, do NOT
import anything): pandas as pd, numpy as np. No print(). Only use exact
column names from the list above.
"""


def build_followup_answer_prompt(insights_text: str, prior_qa_text: str, computed_facts: list[str], question: str) -> str:
    """Second step: compose the actual answer. Grounded in three things,
    in order of trust — real sandbox-computed values (if step one
    produced any — now possibly several, one per angle explored), the
    insights already shown to the user, and nothing else. Explicitly
    told to synthesize across whatever angles were computed rather than
    just restating them, while still saying plainly what it doesn't
    know rather than filling the gap with something plausible-sounding."""
    computed_block = ""
    if computed_facts:
        facts_joined = "\n\n".join(f"Angle {i+1}:\n{f}" for i, f in enumerate(computed_facts))
        computed_block = f"\nTo help answer this, the following was just computed from the real data:\n{facts_joined}\n"
    return f"""You are answering a follow-up question about a business
analysis you already ran, in a short conversational reply — 3-5
sentences unless the question genuinely needs more.

Business insights already identified:
{insights_text}
{prior_qa_text}{computed_block}
User's follow-up question: {question}

Answer using ONLY the information above. If several angles were
computed, synthesize across them — say what they do and don't explain,
the way an analyst would ("the data shows X, which suggests Y, but
doesn't tell us Z"). Do not invent a number, percentage, trend, or fact
that isn't shown above. If none of the computed angles turned out
informative, say plainly what the data couldn't show and what
additional data (pricing strategy, marketing spend, inventory levels,
etc.) would help answer it — but only after actually reporting what was
checked, not instead of checking.
"""


def build_explain_prompt(insight: str, user_query: str) -> str:
    return f"""You are a senior business analyst explaining a finding to a decision-maker.

Insight to explain: {insight}

This came from a dataset analysis where the user asked: "{user_query}"

Explain this in business terms. Cover exactly these four points:

1. Why this pattern likely exists — what business dynamics or behaviours cause it
2. What the business risk or opportunity is — quantify if possible
3. What a decision-maker should do — be specific about the action, team, or budget involved
4. What could make this insight wrong — data limitations, seasonality, confounders

Be direct. No filler. Plain English. No mentions of data cleaning or data structure.
"""
