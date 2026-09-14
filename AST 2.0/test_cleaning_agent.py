"""
Functional proof for cleaning_agent.py (the data quality agent).

WHAT THIS CAN TEST HERE
    - build_defect_profile / format_defect_profile_for_llm (profiler.py,
      pure functions, no streamlit dependency at the point they're called
      here since they only touch pandas/numpy)
    - _parse_defects and _parse_fix_response (cleaning_agent.py, pure
      string parsing — including the realistic-formatting variants that
      broke the FIRST version of this parsing logic in production:
      markdown bold, missing numbering, lowercase labels)
    - A full mechanical dry-run of the sandboxed execution path: a
      hand-written fix response in the EXACT format the LLM is asked to
      produce, parsed, then actually run through sandbox_limits.py —
      proves the plumbing (parsing + sandboxed exec of agent-authored
      fix code) works correctly end-to-end, without needing a real LLM
      call.

WHAT THIS CANNOT TEST HERE (and why)
    run_cleaning_agent()'s actual LLM reasoning steps — it lazily
    imports query_engine.py, which needs llm_provider.py, which needs
    streamlit. Same limitation as every other LLM-calling path in this
    project. Unlike the OLD version of this agent, there is no
    "no nulls remain → skip the LLM entirely" fast path to test here
    instead: the diagnosis step now always runs, on purpose, because
    real data quality defects (inconsistent category spellings,
    implausible values) can exist on a dataset with zero missing
    values — gating the whole agent behind "are there nulls?" would
    make it blind to exactly the class of defect it exists to catch.
    RUN THIS ON YOUR MACHINE for a real end-to-end run including actual
    model reasoning.

Run: python3 test_cleaning_agent.py
"""

import pandas as pd

from cleaning_agent import _parse_defects, _parse_fix_response
from profiler import build_defect_profile, format_defect_profile_for_llm
from sandbox_limits import run_sandboxed

print("=" * 70)
print("PART 1 — widened defect profile (no column cap, full low-cardinality values)")
print("=" * 70)

df = pd.DataFrame({
    "customer_id": [1, 2, 3, 4, 5, 6],
    "country": ["USA", "U.S.", "United States", "Canada", "USA", "Canada"],
    "order_value": [100.0, 250.0, None, 80.0, 300.0, 5000.0],  # 5000 is a plausible outlier
    "region": ["West", "East", "West", "East", "West", "East"],
})

profile = build_defect_profile(df)
assert "country" in profile["categorical_values"], "every categorical column should appear, no 6-column cap"
country_values = profile["categorical_values"]["country"]["all_distinct_values"]
assert set(country_values.keys()) == {"USA", "U.S.", "United States", "Canada"}, \
    "ALL distinct values should be visible for a low-cardinality column, not just top-5"
print(f"\ncountry column values seen by the profile: {country_values}")
print("[OK] widened profile shows every column and every distinct value for low-cardinality columns")

profile_text = format_defect_profile_for_llm(profile)
assert "USA" in profile_text and "U.S." in profile_text
print("[OK] formatted profile text includes the spelling-inconsistency evidence")

print("\n" + "=" * 70)
print("PART 2 — parsing a defect diagnosis (including messy real-world formatting)")
print("=" * 70)

# The FIRST version of this parser (and the sibling one in
# auto_analyze_service.py) silently returned zero results the first
# time it met real, rather than hand-written, LLM output — because it
# required an exact numbered "CHART N:"-style header. These variants
# are exactly the kind of formatting drift that broke it.
variants = {
    "plain": (
        "COLUMN: country\n"
        "TYPE: inconsistent_categories\n"
        "EVIDENCE: values include \"USA\", \"U.S.\", \"United States\" for one country\n"
        "ACTION: map all variants to \"United States\"\n"
    ),
    "markdown bold": (
        "**COLUMN:** country\n"
        "**TYPE:** inconsistent_categories\n"
        "**EVIDENCE:** USA / U.S. / United States all appear\n"
        "**ACTION:** normalize spellings\n"
    ),
    "lowercase": (
        "column: order_value\n"
        "type: outlier\n"
        "evidence: 5000 is far above the other values (100-300 range)\n"
        "action: flag or cap the outlier\n"
    ),
    "no defects": "After review, NO DEFECTS FOUND.",
}

for name, raw in variants.items():
    defects = _parse_defects(raw)
    print(f"  {name}: {len(defects)} defect(s) parsed")
    if "no defects" in name:
        assert defects == [], "'NO DEFECTS FOUND' should parse to an empty list"
    else:
        assert len(defects) == 1, f"expected 1 defect for variant '{name}', got {len(defects)}"
        # snake_case values must survive parsing — an earlier version of
        # this parser stripped underscores as "markdown", corrupting
        # inconsistent_categories into inconsistentcategories.
        assert "_" in defects[0]["type"] or defects[0]["type"] == "outlier", \
            f"defect type should keep its underscore: {defects[0]['type']}"
print("[OK] parser handles plain/markdown/lowercase formatting and preserves snake_case values")

print("\n" + "=" * 70)
print("PART 3 — parsing a realistic fix response")
print("=" * 70)

fake_fix_response = """REASONING:
- country: normalized USA/U.S./United States to one canonical value — fixes the inconsistent category
- order_value: filled the remaining null with the column median — plausible random missingness

```python
cleaned_df = df.copy()
cleaned_df["country"] = cleaned_df["country"].replace({"USA": "United States", "U.S.": "United States"})
cleaned_df["order_value"] = cleaned_df["order_value"].fillna(cleaned_df["order_value"].median())
```
"""

reasoning, code = _parse_fix_response(fake_fix_response)
print(f"\nreasoning: {reasoning}")
print(f"code:\n{code}")
assert len(reasoning) == 2
assert "cleaned_df" in code
print("[OK] correctly splits reasoning from code")

print("\n" + "=" * 70)
print("PART 4 — the parsed fix code actually runs through the real sandbox")
print("=" * 70)

sbr = run_sandboxed(code, df, include_charts=False)
print(f"\nsuccess={sbr.success} produced={sbr.produced}")
if sbr.success:
    print(sbr.value)
assert sbr.success, f"agent-style fix code should run cleanly through the sandbox: {sbr.error}"
assert sbr.value["country"].nunique() == 2, "USA/U.S./United States should have collapsed into one value, leaving United States + Canada"
assert sbr.value["order_value"].isnull().sum() == 0
print("[OK] agent-authored fix code runs correctly through the real sandboxed pipeline")

print("\n" + "=" * 70)
print("PART 5 — real end-to-end run (run on your machine)")
print("=" * 70)
print("On your machine, with GROQ_API_KEY set, try something like:")
print("""
    from cleaning_agent import run_cleaning_agent
    with open("sales_data_sample.csv", "rb") as f:
        result = run_cleaning_agent(f.read(), "sales_data_sample.csv", "session-1")
    print(result["defects"])     # real diagnosed defects, with evidence
    print(result["reasoning"])   # real per-defect fix reasoning from the model
    print(result["after"])       # should show nullValues == 0
""")

print("=" * 70)
print("Done.")
