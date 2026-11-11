import numpy as np
import pandas as pd

from core.detectors import detect_all
from core.executor import Executor
from core.profiler import profile_dataframe
from core.rules import (OP_PRIORITY, build_proposals, fallback_operation, order_operations,
                        quick_options, resolve, rules_only_proposals)
from core.schemas import Operation
from tests.helpers import make_op, messy_frame


def prep(df):
    prof = profile_dataframe(df)
    return prof, detect_all(df, prof)


def op_for(df, issue_id):
    prof, issues = prep(df)
    out = resolve(issues, prof)
    return out.resolved.get(issue_id), out


def nulls_frame(n, null_n, values):
    col = list(values) + [None] * null_n
    col = (col * 3)[: n]
    return pd.DataFrame({"c": col[: n - null_n] + [None] * null_n})


# ---- null rules ------------------------------------------------------------------------------------
def test_numeric_nulls_get_median_with_confidence_tiers():
    rng = np.random.default_rng(0)
    for null_pct, conf in ((3, "high"), (15, "medium"), (45, "low")):
        x = rng.normal(0, 1, 200)
        x[: int(2 * null_pct)] = np.nan
        op, _ = op_for(pd.DataFrame({"x": x}), "nulls:x")
        assert op.op == "impute" and op.params["strategy"] == "median" and op.confidence == conf, null_pct


def test_skewed_numeric_also_gets_median_never_mean():
    x = np.append(np.random.default_rng(0).lognormal(0, 2, 195), [np.nan] * 5)
    op, _ = op_for(pd.DataFrame({"x": x}), "nulls:x")
    assert op.params["strategy"] == "median"


def test_categorical_nulls_get_mode_and_bool_too():
    df = pd.DataFrame({"c": ["a", "b", "a", None] * 25, "flag": pd.array([True, False, None, True] * 25, dtype="boolean")})
    assert op_for(df, "nulls:c")[0].params["strategy"] == "mode"
    assert op_for(df, "nulls:flag")[0].params["strategy"] == "mode"


def test_more_than_60_percent_nulls_drops_the_column_and_supersedes_other_issues():
    df = pd.DataFrame({"mostly_empty": [" a "] * 30 + [None] * 70, "ok": range(100)})
    op, out = op_for(df, "nulls:mostly_empty")
    assert op.op == "drop_column" and op.column == "mostly_empty"
    assert "whitespace:mostly_empty" not in out.resolved
    assert any(i.id == "whitespace:mostly_empty" for i in out.superseded)


def test_free_text_dates_and_id_nulls_are_left_to_the_llm():
    n = 100
    df = pd.DataFrame({
        "note": [f"free text number {i}" for i in range(n - 5)] + [None] * 5,
        "uid": [float(i) for i in range(n - 5)] + [None] * 5,
        "ts": list(pd.date_range("2024-01-01", periods=n - 5)) + [pd.NaT] * 5,
    })
    prof, issues = prep(df)
    out = resolve(issues, prof)
    assert {i.id for i in out.unresolved} == {"nulls:note", "nulls:uid", "nulls:ts"}


def test_nulls_in_a_text_number_column_are_imputed_after_its_cast():
    df = pd.DataFrame({"score": [str(i % 10) for i in range(40)] + [None] * 4, "k": range(44)})  # repeats -> not an ID
    prof, issues = prep(df)
    props = rules_only_proposals(issues, prof)
    assert [p.operation.op for p in props] == ["cast_type", "impute"]
    ex = Executor(df)
    for p in props:
        ex.apply(p.operation, approved_by="user")
    assert ex.current["score"].isna().sum() == 0


# ---- other rules ---------------------------------------------------------------------------------------
def test_duplicates_whitespace_case_dates_numbers():
    df = messy_frame()
    assert op_for(df, "exact_duplicates")[0].op == "drop_duplicates"
    assert op_for(df, "whitespace:city")[0].op == "strip_whitespace"
    case = op_for(df, "case_inconsistency:country")[0]
    assert case.op == "normalize_case" and case.params["mode"] in {"lower", "upper", "title"}
    dates = op_for(df, "date_like_strings:joined")[0]
    assert dates.op == "parse_dates" and dates.params["format"] == "%d/%m/%Y" and dates.confidence == "high"
    assert op_for(df, "mixed_types:score")[0].params["dtype"] == "int"


def test_identifier_like_text_numbers_are_never_cast():
    df = pd.DataFrame({"zip": ["01234", "02345", "10001", "20002", "30003", "40004", "50005", "60006"] * 3})
    op, out = op_for(df, "mixed_types:zip")
    assert op is None and any(i.id == "mixed_types:zip" for i in out.unresolved)
    assert fallback_operation(out.unresolved[0], profile_dataframe(df)) is None


def test_ambiguous_dates_are_unresolved_with_low_confidence_fallback():
    df = pd.DataFrame({"d": ["01/02/2024", "03/04/2024", "05/06/2024", "07/08/2024", "09/10/2024"]})
    prof, issues = prep(df)
    out = resolve(issues, prof)
    assert [i.id for i in out.unresolved] == ["date_like_strings:d"]
    fb = fallback_operation(out.unresolved[0], prof)
    assert fb.op == "parse_dates" and fb.confidence == "low" and fb.source == "rule"


def test_dates_with_leftovers_get_medium_confidence_and_a_warning_in_the_rationale():
    df = pd.DataFrame({"d": ["2024-01-0%d" % i for i in range(1, 10)] + ["garbage"]})
    op, _ = op_for(df, "date_like_strings:d")
    assert op.confidence == "medium" and "1 value(s)" in op.rationale


# ---- fallbacks ---------------------------------------------------------------------------------------------
def test_every_fallback_is_low_confidence_rule_sourced_and_valid_json():
    df = messy_frame()
    prof, issues = prep(df)
    out = resolve(issues, prof)
    seen = set()
    for issue in out.unresolved:
        fb = fallback_operation(issue, prof)
        if fb is None:
            continue
        seen.add(issue.type)
        assert fb.confidence == "low" and fb.source == "rule"
        assert Operation.model_validate_json(fb.model_dump_json()) == fb
    assert {"outliers", "constant_column"} <= seen


def test_near_duplicate_fallback_maps_variants_to_canonical():
    df = pd.DataFrame({"s": ["California"] * 8 + ["Californa"] * 2 + ["Texas"] * 5})
    prof, issues = prep(df)
    issue = next(i for i in issues if i.type == "near_duplicate_categories")
    fb = fallback_operation(issue, prof)
    assert fb.op == "map_values" and fb.params["mapping"] == {"Californa": "California"}


# ---- quick options ----------------------------------------------------------------------------------------------
def test_quick_options_for_numeric_and_text_null_issues():
    df = pd.DataFrame({"n": [1.0, None, 3.0, 4.0] * 10, "c": ["a", None, "b", "a"] * 10})
    prof, issues = prep(df)
    num = quick_options(next(i for i in issues if i.id == "nulls:n"), prof)
    txt = quick_options(next(i for i in issues if i.id == "nulls:c"), prof)
    assert set(num) == {"Median", "Mean", "Mode", "Drop rows"} and set(txt) == {"Mode", "Drop rows"}
    assert all(o.source == "user" for o in num.values())
    assert num["Drop rows"].op == "drop_rows_where_null" and num["Drop rows"].params["subset"] == ["n"]
    assert quick_options(next(i for i in detect_all(messy_frame()) if i.type == "exact_duplicates"), prof) == {}


# ---- ordering & end-to-end ---------------------------------------------------------------------------------------
def test_order_operations_is_a_stable_dependency_sort():
    ops = [make_op("drop_column", "a"), make_op("impute", "b", strategy="mode"), make_op("strip_whitespace", "c"),
           make_op("cast_type", "d", dtype="int"), make_op("strip_whitespace", "e")]
    ordered = [(o.op, o.column) for o in order_operations(ops)]
    assert ordered == [("strip_whitespace", "c"), ("strip_whitespace", "e"), ("cast_type", "d"),
                       ("impute", "b"), ("drop_column", "a")]
    assert OP_PRIORITY["strip_whitespace"] < OP_PRIORITY["map_values"] < OP_PRIORITY["normalize_case"] \
        < OP_PRIORITY["parse_dates"] < OP_PRIORITY["cast_type"] < OP_PRIORITY["drop_duplicates"] < OP_PRIORITY["impute"]


def test_llm_answers_override_fallbacks_but_never_rule_results():
    df = messy_frame()
    prof, issues = prep(df)
    llm = {"outliers:income": make_op("clip_outliers", "income", method="manual", bounds=[0, 100000]),
           "exact_duplicates": make_op("drop_column", "id")}  # must be ignored: rules already resolved it
    props = {p.id: p for p in build_proposals(issues, prof, llm_ops=llm)}
    assert props["outliers:income"].operation.params["method"] == "manual"
    assert props["exact_duplicates"].operation.op == "drop_duplicates"


def test_rules_only_mode_runs_end_to_end_with_no_failures():
    """Core principle 4: with every LLM disabled the app still produces a clean file."""
    df = messy_frame()
    prof, issues = prep(df)
    props = rules_only_proposals(issues, prof)
    assert all(p.operation is not None for p in props)
    ex = Executor(df)
    for p in props:
        assert ex.preview(p.operation).ok, (p.id, ex.preview(p.operation).error)
        ex.apply(p.operation, confirm_large_drop=True, approved_by="user")
    cleaned = ex.current
    assert "const" not in cleaned.columns and cleaned["age"].notna().all() and cleaned["city"].notna().all()
    assert str(cleaned["joined"].dtype).startswith("datetime") and not cleaned.duplicated().any()
    assert not (cleaned["city"].str.startswith(" ") | cleaned["city"].str.endswith(" ")).any()
    # second pass finds only the issues the user chose to leave (none of the rule-resolvable ones)
    left = {i.type for i in detect_all(cleaned)}
    assert not left & {"exact_duplicates", "whitespace", "nulls", "case_inconsistency", "date_like_strings"}


def test_rules_are_deterministic_and_fast_path_uses_no_llm_module():
    df = messy_frame()
    prof, issues = prep(df)
    a = [p.model_dump() for p in rules_only_proposals(issues, prof)]
    b = [p.model_dump() for p in rules_only_proposals(*(lambda p: (detect_all(df, p), p))(profile_dataframe(df.copy())))]
    assert a == b
