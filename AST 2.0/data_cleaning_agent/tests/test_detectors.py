import numpy as np
import pandas as pd

from core import detectors as d
from core.detectors import detect_all
from core.profiler import profile_dataframe
from tests.helpers import messy_frame


def run(fn, df):
    return fn(df, profile_dataframe(df))


def one(issues, type_, col=None):
    found = [i for i in issues if i.type == type_ and i.column == col]
    assert len(found) == 1, [i.id for i in issues]
    return found[0]


# ---- nulls -------------------------------------------------------------------------------
def test_nulls_include_blank_strings_and_report_evidence():
    df = pd.DataFrame({"a": [1.0, None, 3.0, 4.0], "s": ["x", " ", None, "y"], "ok": [1, 2, 3, 4]})
    issues = run(d.detect_nulls, df)
    assert {i.column for i in issues} == {"a", "s"}
    s = one(issues, "nulls", "s")
    assert s.affected_rows == 2 and s.affected_pct == 50.0 and s.evidence["blank_strings"] == 1


# ---- duplicates -----------------------------------------------------------------------------
def test_exact_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    i = one(run(d.detect_exact_duplicates, df), "exact_duplicates")
    assert i.affected_rows == 1 and i.id == "exact_duplicates"
    assert run(d.detect_exact_duplicates, pd.DataFrame({"a": [1, 2]})) == []


# ---- mixed types -----------------------------------------------------------------------------------
def test_numeric_stored_as_text_and_identifier_guard():
    df = pd.DataFrame({
        "score": [str(i) for i in range(10)],
        "zip": ["01234", "02345", "10001", "20002", "30003", "40004", "50005", "60006", "70007", "80008"],
        "amount": ["1.5", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "phone_no": [str(9000000000 + i) for i in range(10)],
        "customer_id": [str(1000 + i) for i in range(10)],
    })
    issues = run(d.detect_mixed_types, df)
    assert one(issues, "mixed_types", "score").evidence["identifier_like"] is False
    assert one(issues, "mixed_types", "score").evidence["suggested_dtype"] == "int"
    assert one(issues, "mixed_types", "amount").evidence["suggested_dtype"] == "float"
    assert one(issues, "mixed_types", "zip").evidence["leading_zeros"] is True
    assert one(issues, "mixed_types", "zip").evidence["identifier_like"] is True
    assert one(issues, "mixed_types", "customer_id").evidence["identifier_like"] is True  # name hint


def test_numeric_with_garbage_and_plain_text_not_flagged():
    df = pd.DataFrame({
        "q": ["1", "2", "3", "4", "5", "6", "N/A", "n/a", "7", "8"],
        "names": ["ann", "bob", "cy", "dee", "eve", "fay", "gus", "hal", "ian", "joy"],
        "mostly_text": ["a", "b", "c", "d", "e", "f", "g", "h", "1", "i"],
    })
    issues = run(d.detect_mixed_types, df)
    q = one(issues, "mixed_types", "q")
    assert q.evidence["subtype"] == "numeric_with_garbage" and q.affected_rows == 2
    assert set(q.evidence["bad_values"]) == {"N/A", "n/a"}
    assert {i.column for i in issues} == {"q"}


# ---- dates ---------------------------------------------------------------------------------------------
def test_unambiguous_dates_are_detected_with_their_format():
    df = pd.DataFrame({"d": ["25/01/2024", "13/02/2024", "01/03/2024", "20/04/2024", "21/05/2024"]})
    i = one(run(d.detect_date_issues, df), "date_like_strings", "d")
    assert i.evidence["format"] == "%d/%m/%Y" and i.evidence["ambiguous"] is False
    assert i.evidence["unparseable_count"] == 0


def test_ambiguous_dates_flagged_day_first_assumed():
    df = pd.DataFrame({"d": ["01/02/2024", "03/04/2024", "05/06/2024", "07/08/2024", "09/10/2024"]})
    i = one(run(d.detect_date_issues, df), "date_like_strings", "d")
    assert i.evidence["ambiguous"] is True and i.evidence["format"] == "%d/%m/%Y"
    assert "%m/%d/%Y" in i.evidence["alternatives"] and i.evidence["assumed"] == "day-first"


def test_iso_dates_and_unparseable_leftovers():
    good = ["2024-01-0%d" % i for i in range(1, 10)]
    df = pd.DataFrame({"d": good + ["2024-13-45", "not a date"]})
    issues = run(d.detect_date_issues, df)
    main = one(issues, "date_like_strings", "d")
    assert main.evidence["format"] == "%Y-%m-%d" and main.evidence["unparseable_count"] == 2
    bad = one(issues, "unparseable_dates", "d")
    assert bad.affected_rows == 2 and "'not a date'" in bad.samples


def test_month_name_dates_and_non_dates():
    df = pd.DataFrame({
        "d": ["05 Jan 2024", "06 Feb 2024", "07 Mar 2024", "08 Apr 2024", "09 May 2024"],
        "n": ["abc", "def", "ghi", "jkl", "mno"],
        "v": ["1.2.3", "4.5.6", "7.8.9", "1.1.1", "2.2.2"],  # version-ish, not dates
    })
    issues = run(d.detect_date_issues, df)
    assert one(issues, "date_like_strings", "d").evidence["format"] == "%d %b %Y"
    assert {i.column for i in issues} == {"d"}


# ---- whitespace & case ---------------------------------------------------------------------------------
def test_whitespace_detected_only_where_present():
    df = pd.DataFrame({"a": [" x", "y", "z  "], "b": ["x", "y", "z"], "c": ["x  y", "y", "z"]})
    issues = run(d.detect_whitespace, df)
    assert {i.column for i in issues} == {"a", "c"}
    assert one(issues, "whitespace", "a").evidence["leading_trailing"] == 2
    assert one(issues, "whitespace", "c").evidence["internal_runs"] == 1


def test_case_inconsistency_only_for_categorical_columns():
    cat = ["usa", "USA", "Usa", "india", "India"] * 6
    free = [f"Free Text Sentence {i}" if i % 2 else f"free text sentence {i}" for i in range(30)]
    df = pd.DataFrame({"country": cat, "free": free})
    issues = run(d.detect_case_inconsistency, df)
    i = one(issues, "case_inconsistency", "country")
    assert i.evidence["clashing_groups"] == 2 and i.evidence["dominant_style"] in {"lower", "upper", "title"}
    assert {x.column for x in issues} == {"country"}


# ---- outliers ---------------------------------------------------------------------------------------------
def test_outliers_iqr_and_skips_ids_flags_small_samples():
    rng = np.random.default_rng(1)
    x = rng.normal(100, 5, 60)
    x[0], x[1] = 900, -400
    df = pd.DataFrame({"x": x, "uid": range(60), "flag": [0, 1] * 30, "tiny": [1.0, 2, 3, 100] + [None] * 56})
    issues = run(d.detect_outliers, df)
    i = one(issues, "outliers", "x")
    assert i.affected_rows >= 2 and i.evidence["n_high"] >= 1 and i.evidence["n_low"] >= 1
    assert {x.column for x in issues} == {"x"}


def test_no_outlier_issue_when_iqr_is_zero_or_none_outside():
    df = pd.DataFrame({"flat": [5.0] * 29 + [500.0], "norm": np.linspace(0, 1, 30)})
    assert run(d.detect_outliers, df) == []


# ---- constants ----------------------------------------------------------------------------------------------
def test_constant_columns():
    df = pd.DataFrame({"c": ["x", "x", None], "v": [1, 2, 3], "n": [None, None, None]})
    issues = run(d.detect_constant_columns, df)
    assert [i.column for i in issues] == ["c"]


# ---- near duplicates -------------------------------------------------------------------------------------------
def test_near_duplicates_cluster_typos_and_punctuation_not_case_or_digits():
    vals = (["California"] * 8 + ["Californa"] * 2 + ["New York"] * 8 + ["new york"] * 2
            + ["U.S.A"] * 3 + ["USA"] * 6 + ["Group 1"] * 4 + ["Group 2"] * 4 + ["Texas"] * 5)
    df = pd.DataFrame({"state": vals})
    i = one(run(d.detect_near_duplicate_categories, df), "near_duplicate_categories", "state")
    clusters = {c["canonical"]: {v["value"] for v in c["variants"]} for c in i.evidence["clusters"]}
    assert clusters["California"] == {"Californa"}
    assert clusters["USA"] == {"U.S.A"}
    assert "New York" not in clusters and "Group 1" not in clusters and "Group 2" not in clusters
    assert i.affected_rows == 2 + 3
    assert "Californa -> California" in i.samples


def test_near_duplicates_canonical_tie_break_is_alphabetical():
    df = pd.DataFrame({"c": ["Colour", "Color"] * 5 + ["Other"] * 4})
    i = one(run(d.detect_near_duplicate_categories, df), "near_duplicate_categories", "c")
    assert i.evidence["clusters"][0]["canonical"] == "Color"


# ---- orchestration -----------------------------------------------------------------------------------------------
def test_detect_all_finds_every_injected_defect_and_is_deterministic():
    df = messy_frame()
    a, b = detect_all(df), detect_all(df.copy())
    assert [i.model_dump() for i in a] == [i.model_dump() for i in b]
    ids = {i.id for i in a}
    assert {"exact_duplicates", "whitespace:city", "case_inconsistency:country", "date_like_strings:joined",
            "mixed_types:score", "nulls:age", "nulls:city", "outliers:income", "constant_column:const"} <= ids
    assert all(len(i.samples) <= 8 for i in a)


def test_detect_all_on_clean_data_finds_nothing():
    df = pd.DataFrame({"a": np.linspace(0, 1, 40), "b": ["x", "y"] * 20})
    assert detect_all(df) == []
