import json

import pytest
from pydantic import ValidationError

from core.schemas import AuditEntry, Operation, OP_NAMES
from tests.helpers import make_op


def test_closed_library_has_exactly_the_ten_ops():
    assert set(OP_NAMES) == {
        "drop_duplicates", "impute", "cast_type", "parse_dates", "strip_whitespace",
        "normalize_case", "clip_outliers", "map_values", "drop_column", "drop_rows_where_null",
    }


def test_unknown_op_rejected():
    with pytest.raises(ValidationError):
        Operation(op="run_python", column="a", params={})


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        Operation(op="drop_column", column="a", params={}, code="os.system('x')")


def test_unknown_param_rejected():
    with pytest.raises(ValidationError, match="Extra inputs|extra"):
        make_op("impute", "a", strategy="median", evil=1)


def test_column_ops_require_a_column():
    with pytest.raises(ValidationError, match="needs a column"):
        make_op("normalize_case", None, mode="lower")


def test_row_ops_reject_a_column():
    with pytest.raises(ValidationError, match="rows"):
        make_op("drop_duplicates", "a")


@pytest.mark.parametrize("params", [{"strategy": "constant"}, {"strategy": "median", "value": 1}, {"strategy": "mode2"}])
def test_impute_param_rules(params):
    with pytest.raises(ValidationError):
        make_op("impute", "a", **params)


@pytest.mark.parametrize("params", [
    {"method": "iqr", "bounds": [0, 1]},
    {"method": "percentile"},
    {"method": "percentile", "bounds": [50, 10]},
    {"method": "manual"},
    {"method": "manual", "bounds": [5, 1]},
    {"method": "iqr", "k": 0},
])
def test_clip_param_rules(params):
    with pytest.raises(ValidationError):
        make_op("clip_outliers", "a", **params)


def test_map_values_must_not_be_empty_or_huge():
    with pytest.raises(ValidationError):
        make_op("map_values", "a", mapping={})
    with pytest.raises(ValidationError):
        make_op("map_values", "a", mapping={str(i): "x" for i in range(1001)})


def test_parse_dates_format_must_be_strftime():
    with pytest.raises(ValidationError):
        make_op("parse_dates", "a", format="YYYY-MM-DD")


def test_drop_rows_needs_subset():
    with pytest.raises(ValidationError):
        make_op("drop_rows_where_null", None, subset=[])


def test_defaults_are_filled_in_and_json_safe():
    op = make_op("clip_outliers", "a", method="manual", bounds=(0, 10))
    assert op.params == {"method": "manual", "k": 1.5, "bounds": [0.0, 10.0]}
    json.dumps(op.model_dump())  # must not raise


def test_rationale_is_collapsed_and_capped():
    op = Operation(op="drop_column", column="a", rationale="line one\n\n  line   two " + "x" * 500)
    assert "\n" not in op.rationale and "  " not in op.rationale
    assert len(op.rationale) <= 300


def test_key_ignores_rationale_confidence_and_source():
    a = Operation(op="impute", column="a", params={"strategy": "median"}, rationale="x", confidence="low")
    b = Operation(op="impute", column="a", params={"strategy": "median"}, rationale="y", confidence="high", source="user")
    c = make_op("impute", "a", strategy="mean")
    assert a.key() == b.key() != c.key()


def test_round_trip_through_json_revalidates():
    op = make_op("map_values", "c", mapping={"a": "b"})
    again = Operation.model_validate_json(op.model_dump_json())
    assert again == op


def test_audit_entry_forbids_extra_fields():
    with pytest.raises(ValidationError):
        AuditEntry(seq=1, timestamp="t", action="apply", operation=make_op("drop_column", "a"),
                   rows_before=1, rows_after=1, rows_affected=0, source="rule", approved_by="u", sneaky=1)
