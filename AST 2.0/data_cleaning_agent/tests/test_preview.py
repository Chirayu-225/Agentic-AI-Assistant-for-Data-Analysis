import pandas as pd

from core import operations
from core.preview import preview
from tests.helpers import make_op


def frame():
    return pd.DataFrame({"n": [1.0, None, 3.0, None, 5.0], "s": [" a ", "b", "b", "c", None]})


def test_value_op_preview_counts_and_samples():
    df = frame()
    before = df.copy(deep=True)
    pv = preview(df, make_op("impute", "n", strategy="median"))
    assert pv.ok and pv.rows_affected == 2 and pv.pct_affected == 40.0 and pv.rows_dropped == 0
    assert pv.rows_before == pv.rows_after == 5 and not pv.requires_extra_confirmation
    assert {(s.row, s.before, s.after) for s in pv.samples} == {("1", "<null>", "3"), ("3", "<null>", "3")}
    pd.testing.assert_frame_equal(df, before)  # a preview never changes the data


def test_strip_samples_show_the_quotes_so_spaces_are_visible():
    pv = preview(frame(), make_op("strip_whitespace", "s"))
    assert pv.samples[0].before == "' a '" and pv.samples[0].after == "'a'"


def test_row_drop_preview_shows_dropped_rows():
    pv = preview(frame(), make_op("drop_rows_where_null", None, subset=["n"]))
    assert pv.rows_dropped == 2 and pv.rows_after == 3 and pv.pct_dropped == 40.0
    assert all(s.after == "(row removed)" and s.column is None for s in pv.samples)
    assert pv.samples[0].before.startswith("n=<null>")


def test_extra_confirmation_flag_uses_the_5_percent_rule():
    df = pd.DataFrame({"a": [None] + [1.0] * 19})           # 1/20 = 5% exactly -> allowed
    assert not preview(df, make_op("drop_rows_where_null", None, subset=["a"])).requires_extra_confirmation
    df = pd.DataFrame({"a": [None, None] + [1.0] * 18})     # 10% -> needs confirmation
    assert preview(df, make_op("drop_rows_where_null", None, subset=["a"])).requires_extra_confirmation
    assert preview(df, make_op("drop_rows_where_null", None, subset=["a"]), row_drop_threshold=0.5).requires_extra_confirmation is False
    assert operations.ROW_DROP_CONFIRM_THRESHOLD == 0.05


def test_bad_operation_returns_an_error_instead_of_raising():
    pv = preview(frame(), make_op("drop_column", "ghost"))
    assert pv.ok is False and "ghost" in pv.error and pv.rows_after == 5
    pv = preview(frame(), make_op("impute", "s", strategy="mean"))
    assert pv.ok is False and "numeric" in pv.error


def test_warnings_for_noops_drops_and_lossy_conversions():
    assert "would not change any rows" in " ".join(preview(frame(), make_op("map_values", "s", mapping={"zz": "y"})).warnings)
    assert "Removes the whole 's' column" in " ".join(preview(frame(), make_op("drop_column", "s")).warnings)
    dates = pd.DataFrame({"d": ["2024-01-01", "oops", "2024-01-03"]})
    w = preview(dates, make_op("parse_dates", "d", format="%Y-%m-%d")).warnings
    assert any("1 value(s) are not valid dates" in x for x in w)
    nums = pd.DataFrame({"t": ["1", "x", "3"]})
    assert any("could not be converted" in x for x in preview(nums, make_op("cast_type", "t", dtype="float")).warnings)


def test_dtype_change_is_reported_and_duplicate_index_is_tolerated():
    pv = preview(pd.DataFrame({"t": ["1", "2"]}), make_op("cast_type", "t", dtype="int"))
    assert pv.dtype_before != pv.dtype_after and pv.dtype_after == "Int64"
    dup = pd.concat([frame(), frame()])  # repeated row labels
    assert preview(dup, make_op("strip_whitespace", "s")).ok


def test_sample_count_is_capped():
    df = pd.DataFrame({"s": [" x"] * 50})
    assert len(preview(df, make_op("strip_whitespace", "s"), n_samples=3).samples) == 3
