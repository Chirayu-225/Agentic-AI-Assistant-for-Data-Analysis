"""
Fail-closed approval: there is no default approver, and every value mapping needs a person.

Before: Executor.apply(op) defaulted to approved_by="user", so a caller that forgot the argument (an auto-apply path, a
future proposer) silently got human-level approval for impute, drop_column, clip_outliers and row drops.
"""
import pandas as pd
import pytest

from core.detectors import detect_all
from core.executor import Executor
from core.policy import PolicyError, restriction
from core.profiler import profile_dataframe
from core.rules import fallback_operation
from tests.helpers import make_op


def frame():
    return pd.DataFrame({"kind": ["a", "A", "b", "b"], "n": [1.0, None, 3.0, 4.0]})


RESTRICTED = [
    make_op("impute", "n", strategy="median"),
    make_op("impute", "n", strategy="constant", value=0),
    make_op("drop_column", "n"),
    make_op("clip_outliers", "n", method="iqr"),
    make_op("drop_duplicates"),
    make_op("drop_rows_where_null", None, subset=["n"]),
]


def mapping(source):
    return make_op("map_values", "kind", mapping={"a": "A"}).model_copy(update={"source": source})


@pytest.mark.parametrize("op", RESTRICTED)
def test_restricted_op_without_an_approver_is_refused_and_changes_nothing(op):
    ex = Executor(frame())
    with pytest.raises(PolicyError, match="no approver was given"):
        ex.apply(op)                      # no approved_by at all: must fail closed
    assert ex.version == 0 and ex.audit_log == []
    pd.testing.assert_frame_equal(ex.current, frame())


@pytest.mark.parametrize("op", RESTRICTED)
def test_restricted_op_is_applied_when_a_person_approves(op):
    ex = Executor(frame(), row_drop_threshold=1.0)
    entry = ex.apply(op, approved_by="user")
    assert entry.approved_by == "user" and entry.policy == restriction(op)


def test_harmless_op_without_an_approver_is_applied_and_recorded_as_unspecified():
    ex = Executor(pd.DataFrame({"kind": ["  a ", "b"]}))
    entry = ex.apply(make_op("strip_whitespace", "kind"))
    assert entry.approved_by == "unspecified" and entry.policy is None
    assert ex.audit_records()[0]["approved_by"] == "unspecified"


@pytest.mark.parametrize("source", ["rule", "llm", "user"])
def test_every_value_mapping_needs_a_person_whoever_proposed_it(source):
    op = mapping(source)
    assert restriction(op) == ("llm-value-mapping" if source == "llm" else "value-mapping")
    ex = Executor(frame())
    with pytest.raises(PolicyError):
        ex.apply(op)
    with pytest.raises(PolicyError):
        ex.apply(op, approved_by="auto")
    assert ex.version == 0 and ex.audit_log == []
    entry = ex.apply(op, approved_by="user")
    assert entry.source == source and entry.approved_by == "user"
    assert list(ex.current["kind"]) == ["A", "A", "b", "b"]


def test_rule_proposed_mappings_only_target_values_already_in_the_column():
    """Rules build a mapping from the column's own values (core/rules.py), so the 'target must exist' rule that the
    executor enforces for model-proposed mappings holds for them by construction. This test pins that."""
    frames = [
        pd.DataFrame({"s": ["California"] * 8 + ["Californa"] * 2 + ["Texas"] * 5}),
        pd.DataFrame({"s": ["New York"] * 9 + ["New Yrok"] * 2 + ["Ohio"] * 6 + ["Ohoi"]}),
    ]
    seen = 0
    for df in frames:
        prof = profile_dataframe(df)
        for issue in detect_all(df, prof):
            op = fallback_operation(issue, prof)
            if op is not None and op.op == "map_values":
                seen += 1
                present = set(df[op.column].dropna())
                assert set(op.params["mapping"].values()) <= present, op.params["mapping"]
    assert seen >= 1
