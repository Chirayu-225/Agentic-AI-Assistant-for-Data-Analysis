import numpy as np
import pandas as pd
import pytest

from core import operations
from core.operations import OperationError, apply, rows_affected
from tests.helpers import make_op


def vals(df, col):
    return df[col].tolist()


# ---- immutability ---------------------------------------------------------------------
CASES = [
    ("drop_duplicates", None, {}),
    ("impute", "n", {"strategy": "median"}),
    ("cast_type", "t", {"dtype": "float"}),
    ("strip_whitespace", "s", {}),
    ("normalize_case", "s", {"mode": "upper"}),
    ("clip_outliers", "n", {"method": "manual", "bounds": [0, 2]}),
    ("map_values", "s", {"mapping": {"a": "z"}}),
    ("drop_column", "n", {}),
    ("drop_rows_where_null", None, {"subset": ["n"]}),
]


@pytest.mark.parametrize("op,col,params", CASES)
def test_apply_never_mutates_its_input(op, col, params):
    df = pd.DataFrame({"n": [1.0, None, 3.0, 3.0], "t": ["1", "2", "x", "4"], "s": [" a", "A ", None, "b"]})
    snapshot = df.copy(deep=True)
    out = apply(df, make_op(op, col, **params))
    pd.testing.assert_frame_equal(df, snapshot)
    assert out is not df


# ---- hallucinated / invalid targets -------------------------------------------------------
def test_unknown_column_is_refused_with_clear_message():
    with pytest.raises(OperationError, match="'nope' does not exist"):
        apply(pd.DataFrame({"a": [1]}), make_op("drop_column", "nope"))


def test_unknown_subset_and_group_by_columns_refused():
    df = pd.DataFrame({"a": [1.0, None]})
    with pytest.raises(OperationError, match="ghost"):
        apply(df, make_op("drop_rows_where_null", None, subset=["ghost"]))
    with pytest.raises(OperationError, match="ghost"):
        apply(df, make_op("impute", "a", strategy="mean", group_by="ghost"))


def test_text_ops_refuse_numeric_columns_and_vice_versa():
    df = pd.DataFrame({"n": [1.0, 2.0], "s": ["a", "b"]})
    for op in (make_op("strip_whitespace", "n"), make_op("normalize_case", "n", mode="lower"),
               make_op("map_values", "n", mapping={"1": "2"}), make_op("impute", "s", strategy="mean"),
               make_op("clip_outliers", "s", method="iqr")):
        with pytest.raises(OperationError):
            apply(df, op)


# ---- drop_duplicates --------------------------------------------------------------------
def test_drop_duplicates_keep_first_and_last_and_subset():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
    assert len(apply(df, make_op("drop_duplicates"))) == 3  # rows differ in b
    assert vals(apply(df, make_op("drop_duplicates", subset=["a"])), "b") == ["x", "z"]
    assert vals(apply(df, make_op("drop_duplicates", subset=["a"], keep="last")), "b") == ["y", "z"]


# ---- impute -------------------------------------------------------------------------------
def test_impute_median_mean_mode_constant_ffill():
    df = pd.DataFrame({"n": [1.0, None, 3.0, 100.0]})
    assert vals(apply(df, make_op("impute", "n", strategy="median")), "n")[1] == 3.0
    assert vals(apply(df, make_op("impute", "n", strategy="mean")), "n")[1] == pytest.approx(104 / 3)
    assert vals(apply(df, make_op("impute", "n", strategy="constant", value=0)), "n")[1] == 0
    assert vals(apply(df, make_op("impute", "n", strategy="ffill")), "n")[1] == 1.0
    cat = pd.DataFrame({"c": ["a", "b", "b", None]})
    assert vals(apply(cat, make_op("impute", "c", strategy="mode")), "c")[3] == "b"


def test_impute_mode_tie_is_deterministic():
    df = pd.DataFrame({"c": ["b", "a", "a", "b", None]})
    assert vals(apply(df, make_op("impute", "c", strategy="mode")), "c")[4] == "a"


def test_impute_treats_blank_strings_as_null():
    df = pd.DataFrame({"c": ["a", "  ", "", "a", "b"]})
    out = apply(df, make_op("impute", "c", strategy="mode"))
    assert vals(out, "c") == ["a", "a", "a", "a", "b"]


def test_impute_keeps_whole_number_columns_whole():
    df = pd.DataFrame({"age": [20.0, None, 30.0, 40.0]})  # how read_csv stores ints with gaps
    out = apply(df, make_op("impute", "age", strategy="median"))
    assert str(out["age"].dtype) == "Int64" and vals(out, "age") == [20, 30, 30, 40]
    frac = apply(pd.DataFrame({"age": [20.0, None, 25.0]}), make_op("impute", "age", strategy="mean"))
    assert vals(frac, "age")[1] == 22.5 and frac["age"].dtype == "float64"


def test_impute_on_nullable_int_column_with_fractional_median_widens_to_float():
    s = pd.array([0, 1, 2, 3, None, 4, 5, 6, 7, 8, 9], dtype="Int64")
    df = pd.DataFrame({"n": s})
    out = apply(df, make_op("impute", "n", strategy="median"))  # median 4.5 cannot live in Int64
    assert out["n"].dtype == "float64" and out["n"].iloc[4] == 4.5
    whole = apply(pd.DataFrame({"n": pd.array([1, None, 3], dtype="Int64")}), make_op("impute", "n", strategy="median"))
    assert str(whole["n"].dtype) == "Int64" and whole["n"].tolist() == [1, 2, 3]


def test_impute_group_by_with_global_fallback():
    df = pd.DataFrame({"g": ["a", "a", "b", "b", "c"], "n": [1.0, None, 10.0, None, None]})
    out = apply(df, make_op("impute", "n", strategy="mean", group_by="g"))
    assert vals(out, "n") == [1.0, 1.0, 10.0, 10.0, 5.5]  # group 'c' has no data -> global mean


def test_impute_constant_type_checked():
    with pytest.raises(OperationError, match="not a number"):
        apply(pd.DataFrame({"n": [1.0, None]}), make_op("impute", "n", strategy="constant", value="abc"))
    out = apply(pd.DataFrame({"s": ["a", None]}), make_op("impute", "s", strategy="constant", value="Unknown"))
    assert vals(out, "s") == ["a", "Unknown"]


def test_impute_all_null_column_and_no_nulls_are_noops():
    df = pd.DataFrame({"n": [None, None], "m": [1.0, 2.0]}).astype("float64")
    assert apply(df, make_op("impute", "n", strategy="median"))["n"].isna().all()  # nothing to fill from
    pd.testing.assert_frame_equal(apply(df, make_op("impute", "m", strategy="median")), df)


def test_impute_categorical_dtype():
    df = pd.DataFrame({"c": pd.Categorical(["a", "b", "b", None])})
    out = apply(df, make_op("impute", "c", strategy="mode"))
    assert vals(out, "c") == ["a", "b", "b", "b"] and str(out["c"].dtype) == "category"


# ---- cast_type ----------------------------------------------------------------------------
def test_cast_to_int_float_str_category():
    df = pd.DataFrame({"t": [" 1", "2 ", None, ""]})
    assert vals(apply(df, make_op("cast_type", "t", dtype="int")), "t")[:2] == [1, 2]
    assert str(apply(df, make_op("cast_type", "t", dtype="int"))["t"].dtype) == "Int64"
    f = apply(df, make_op("cast_type", "t", dtype="float"))
    assert f["t"].dtype == "float64" and f["t"].isna().tolist() == [False, False, True, True]
    assert str(apply(pd.DataFrame({"n": [1, 2]}), make_op("cast_type", "n", dtype="str"))["n"].dtype) in ("string", "str")
    assert str(apply(df, make_op("cast_type", "t", dtype="category"))["t"].dtype) == "category"


def test_cast_coerce_vs_raise_and_non_whole_ints():
    df = pd.DataFrame({"t": ["1", "oops", "3"]})
    assert apply(df, make_op("cast_type", "t", dtype="float"))["t"].isna().tolist() == [False, True, False]
    with pytest.raises(OperationError, match="not numbers"):
        apply(df, make_op("cast_type", "t", dtype="float", errors="raise"))
    with pytest.raises(OperationError, match="non-whole"):
        apply(pd.DataFrame({"t": ["1.5", "2"]}), make_op("cast_type", "t", dtype="int"))
    inf = apply(pd.DataFrame({"t": ["inf", "1"]}), make_op("cast_type", "t", dtype="float"))
    assert inf["t"].isna().tolist() == [True, False]


def test_cast_to_bool():
    df = pd.DataFrame({"b": ["Yes", "no", "TRUE", "0", None]})
    out = apply(df, make_op("cast_type", "b", dtype="bool"))
    assert out["b"].tolist()[:4] == [True, False, True, False] and pd.isna(out["b"].iloc[4])
    with pytest.raises(OperationError):
        apply(pd.DataFrame({"b": ["maybe"]}), make_op("cast_type", "b", dtype="bool", errors="raise"))


# ---- parse_dates -------------------------------------------------------------------------------
def test_parse_dates_with_format_and_bad_values_become_nat():
    df = pd.DataFrame({"d": ["05/01/2024", "31/12/2023", "garbage", None]})
    out = apply(df, make_op("parse_dates", "d", format="%d/%m/%Y"))
    assert out["d"].iloc[0] == pd.Timestamp(2024, 1, 5) and out["d"].iloc[1] == pd.Timestamp(2023, 12, 31)
    assert out["d"].isna().tolist() == [False, False, True, True]


def test_parse_dates_errors_raise_and_total_mismatch_refused():
    with pytest.raises(OperationError, match="not dates"):
        apply(pd.DataFrame({"d": ["2024-01-01", "x"]}), make_op("parse_dates", "d", format="%Y-%m-%d", errors="raise"))
    with pytest.raises(OperationError, match="nothing was changed"):
        apply(pd.DataFrame({"d": ["abc", "def"]}), make_op("parse_dates", "d", format="%Y-%m-%d"))


def test_parse_dates_without_format_uses_dayfirst_flag():
    df = pd.DataFrame({"d": ["05/01/2024", "06/02/2024"]})
    assert apply(df, make_op("parse_dates", "d", dayfirst=True))["d"].iloc[0] == pd.Timestamp(2024, 1, 5)
    assert apply(df, make_op("parse_dates", "d", dayfirst=False))["d"].iloc[0] == pd.Timestamp(2024, 5, 1)


def test_parse_dates_on_datetime_column_is_noop():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    pd.testing.assert_frame_equal(apply(df, make_op("parse_dates", "d")), df)


# ---- strip / case --------------------------------------------------------------------------------
def test_strip_whitespace_collapse_and_blank_to_null():
    df = pd.DataFrame({"s": ["  a   b ", "c", "   ", None]})
    out = apply(df, make_op("strip_whitespace", "s"))
    assert out["s"].iloc[0] == "a b" and out["s"].iloc[1] == "c"
    assert out["s"].isna().tolist() == [False, False, True, True]
    keep = apply(df, make_op("strip_whitespace", "s", collapse=False))
    assert keep["s"].iloc[0] == "a   b"


def test_strip_leaves_non_strings_in_object_columns_alone():
    df = pd.DataFrame({"s": pd.Series([" a ", 5, None], dtype=object)})
    out = apply(df, make_op("strip_whitespace", "s"))
    assert out["s"].tolist()[:2] == ["a", 5]


def test_normalize_case_modes_and_nulls():
    df = pd.DataFrame({"s": ["new YORK", "o'brien", None]})
    assert vals(apply(df, make_op("normalize_case", "s", mode="lower")), "s")[0] == "new york"
    assert vals(apply(df, make_op("normalize_case", "s", mode="upper")), "s")[0] == "NEW YORK"
    title = apply(df, make_op("normalize_case", "s", mode="title"))
    assert title["s"].iloc[0] == "New York" and title["s"].iloc[1] == "O'brien"
    assert pd.isna(title["s"].iloc[2])


# ---- clip_outliers ------------------------------------------------------------------------------------
def test_clip_iqr_percentile_manual():
    df = pd.DataFrame({"n": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 1000]})
    iqr = apply(df, make_op("clip_outliers", "n", method="iqr"))
    q1, q3 = df["n"].quantile([0.25, 0.75])
    assert iqr["n"].max() == pytest.approx(q3 + 1.5 * (q3 - q1))
    pct = apply(df, make_op("clip_outliers", "n", method="percentile", bounds=[0, 90]))
    assert pct["n"].max() == pytest.approx(df["n"].quantile(0.9))
    one_sided = apply(df, make_op("clip_outliers", "n", method="manual", bounds=[None, 10]))
    assert one_sided["n"].max() == 10 and one_sided["n"].min() == 1


def test_clip_keeps_integers_integer_and_refuses_flat_iqr():
    df = pd.DataFrame({"n": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]})
    out = apply(df, make_op("clip_outliers", "n", method="manual", bounds=[None, 10]))
    assert str(out["n"].dtype) == "Int64" or out["n"].dtype.kind in "iu"
    with pytest.raises(OperationError, match="IQR is 0"):
        apply(pd.DataFrame({"n": [5.0] * 9 + [500.0]}), make_op("clip_outliers", "n", method="iqr"))


# ---- map_values ---------------------------------------------------------------------------------------------
def test_map_values_only_touches_listed_values():
    df = pd.DataFrame({"c": ["USA", "U.S.", "India", None]})
    out = apply(df, make_op("map_values", "c", mapping={"U.S.": "USA"}))
    assert out["c"].tolist()[:3] == ["USA", "USA", "India"] and pd.isna(out["c"].iloc[3])


def test_map_values_is_single_pass_and_works_on_categoricals():
    out = apply(pd.DataFrame({"c": ["a", "b"]}), make_op("map_values", "c", mapping={"a": "b", "b": "c"}))
    assert out["c"].tolist() == ["b", "c"]  # a->b is NOT chained on to c
    cat = apply(pd.DataFrame({"c": pd.Categorical(["a", "b"])}), make_op("map_values", "c", mapping={"a": "z"}))
    assert cat["c"].tolist() == ["z", "b"] and str(cat["c"].dtype) == "category"


def test_map_values_with_no_matches_is_a_noop():
    df = pd.DataFrame({"c": ["a"]})
    pd.testing.assert_frame_equal(apply(df, make_op("map_values", "c", mapping={"zzz": "y"})), df)


# ---- drop_column / drop_rows_where_null ------------------------------------------------------------------------
def test_drop_column_and_drop_rows_any_all_and_blank_strings():
    df = pd.DataFrame({"a": [1.0, None, None, 2.0], "b": ["x", "  ", None, None]})
    assert list(apply(df, make_op("drop_column", "a")).columns) == ["b"]
    assert len(apply(df, make_op("drop_rows_where_null", None, subset=["a", "b"]))) == 1      # any
    assert len(apply(df, make_op("drop_rows_where_null", None, subset=["a", "b"], how="all"))) == 2
    assert len(apply(df, make_op("drop_rows_where_null", None, subset=["b"]))) == 1  # '  ' counts as null


# ---- rows_affected ---------------------------------------------------------------------------------------------------
def test_rows_affected_counts_by_kind_of_operation():
    df = pd.DataFrame({"n": [1.0, None, 3.0], "s": [" a", "b", "b"], "t": ["1", "2", None]})
    cases = [
        (make_op("impute", "n", strategy="median"), 1),
        (make_op("strip_whitespace", "s"), 1),
        (make_op("drop_rows_where_null", None, subset=["n"]), 1),
        (make_op("drop_column", "s"), 3),
        (make_op("cast_type", "t", dtype="int"), 2),  # representation changed for the 2 non-null cells
        (make_op("map_values", "s", mapping={"zz": "y"}), 0),
    ]
    for op, expected in cases:
        assert rows_affected(df, apply(df, op), op) == expected, op.describe()


def test_drop_flags():
    assert operations.is_destructive(make_op("drop_column", "a"))
    assert operations.is_row_dropping(make_op("drop_duplicates"))
    assert not operations.is_row_dropping(make_op("drop_column", "a"))
    assert not operations.is_destructive(make_op("strip_whitespace", "a"))


def test_no_op_is_marked_changed_for_null_to_null():
    df = pd.DataFrame({"n": [np.nan, 1.0]})
    assert rows_affected(df, apply(df, make_op("impute", "n", strategy="ffill")), make_op("impute", "n", strategy="ffill")) == 0
