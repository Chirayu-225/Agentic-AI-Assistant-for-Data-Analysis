import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from core.detectors import detect_all
from core.executor import ConfirmationRequired, Executor, GuardrailError
from core.operations import OperationError
from core.profiler import profile_dataframe
from core.rules import rules_only_proposals
from core.schemas import Operation
from tests.helpers import make_op, messy_frame


class Clock:
    def __init__(self):
        self.t = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self):
        self.t = self.t.replace(second=self.t.second + 1)
        return self.t


def frame(n=100):
    return pd.DataFrame({"n": [float(i) for i in range(n)], "s": [" a"] * n})


# ---- snapshots & immutability -------------------------------------------------------------------
def test_original_is_deep_copied_and_cannot_be_altered_through_any_accessor():
    src = frame(10)
    ex = Executor(src)
    src.loc[0, "n"] = 999.0                      # caller mutates their own frame afterwards
    handed_out = ex.original
    handed_out.loc[0, "n"] = -1.0                # ...or the returned "original"
    assert ex.original.loc[0, "n"] == 0.0
    ex.apply(make_op("strip_whitespace", "s"), approved_by="user")
    assert ex.original["s"].iloc[0] == " a" and ex.current["s"].iloc[0] == "a"


def test_versions_advance_and_each_op_sees_the_previous_result():
    ex = Executor(frame(10))
    assert ex.version == 0 and not ex.can_undo
    ex.apply(make_op("strip_whitespace", "s"), approved_by="user")
    ex.apply(make_op("normalize_case", "s", mode="upper"), approved_by="user")
    assert ex.version == 2 and ex.current["s"].iloc[0] == "A"
    assert [o.op for o in ex.applied_operations()] == ["strip_whitespace", "normalize_case"]


def test_row_labels_are_normalised_to_csv_order():
    df = pd.concat([frame(3), frame(3)])  # duplicate labels
    assert Executor(df).current.index.tolist() == list(range(6))


# ---- failure leaves state untouched ---------------------------------------------------------------------
def test_failed_operation_changes_nothing_and_logs_nothing():
    ex = Executor(frame(10))
    with pytest.raises(OperationError):
        ex.apply(make_op("drop_column", "ghost"), approved_by="user")
    with pytest.raises(OperationError):
        ex.apply(make_op("impute", "s", strategy="mean"), approved_by="user")
    assert ex.version == 0 and ex.audit_log == [] and ex.applied_operations() == []


# ---- 5% row-drop guardrail ----------------------------------------------------------------------------------
def test_dropping_more_than_5_percent_needs_explicit_confirmation():
    df = pd.DataFrame({"a": [None] * 10 + [1.0] * 90})  # 10% of rows
    ex = Executor(df)
    op = make_op("drop_rows_where_null", None, subset=["a"])
    with pytest.raises(ConfirmationRequired) as err:
        ex.apply(op, approved_by="user")
    assert isinstance(err.value, GuardrailError) and err.value.rows_dropped == 10
    assert "10.0%" in str(err.value) and ex.version == 0 and ex.audit_log == []
    entry = ex.apply(op, confirm_large_drop=True, approved_by="user")
    assert len(ex.current) == 90 and "confirmed" in entry.note


def test_exactly_5_percent_is_allowed_without_confirmation():
    df = pd.DataFrame({"a": [None] * 5 + [1.0] * 95})
    Executor(df).apply(make_op("drop_rows_where_null", None, subset=["a"]), approved_by="user")


def test_guardrail_applies_to_duplicates_and_threshold_is_configurable():
    df = pd.DataFrame({"a": [1] * 20})
    with pytest.raises(ConfirmationRequired):
        Executor(df).apply(make_op("drop_duplicates"), approved_by="user")
    Executor(df, row_drop_threshold=1.0).apply(make_op("drop_duplicates"), approved_by="user")


def test_non_row_ops_never_trigger_the_row_guard():
    Executor(frame(10)).apply(make_op("drop_column", "s"), approved_by="user")


# ---- undo ---------------------------------------------------------------------------------------------------------------
def test_undo_restores_previous_snapshot_and_keeps_the_audit_trail():
    ex = Executor(frame(10))
    ex.apply(make_op("strip_whitespace", "s"), approved_by="user")
    ex.apply(make_op("drop_column", "n"), approved_by="user")
    undo = ex.undo()
    assert undo.action == "undo" and "reverts #2" in undo.note
    assert list(ex.current.columns) == ["n", "s"] and ex.version == 1
    assert [o.op for o in ex.applied_operations()] == ["strip_whitespace"]
    assert [e.action for e in ex.audit_log] == ["apply", "apply", "undo"]
    ex.undo()
    assert ex.current["s"].iloc[0] == " a" and not ex.can_undo
    assert ex.undo() is None


# ---- audit log -------------------------------------------------------------------------------------------------------------
def test_audit_entries_record_everything_the_spec_requires():
    ex = Executor(frame(10), clock=Clock())
    op = Operation(op="strip_whitespace", column="s", rationale="why", confidence="high", source="llm")
    e = ex.apply(op, approved_by="farrel")
    assert (e.seq, e.action, e.source, e.approved_by) == (1, "apply", "llm", "farrel")
    assert (e.rows_before, e.rows_after, e.rows_affected) == (10, 10, 10)
    assert e.operation.op == "strip_whitespace" and e.operation.params == {"collapse": True}
    assert e.timestamp.startswith("2026-01-01T00:00:") and e.timestamp.endswith("+00:00")
    rec = ex.audit_records()[0]
    assert rec["op"] == "strip_whitespace" and rec["column"] == "s" and json.loads(rec["params"]) == {"collapse": True}
    json.dumps(ex.audit_records())  # JSON-safe


def test_noop_is_logged_with_a_note_and_reject_is_recorded_without_changing_data():
    ex = Executor(frame(10))
    e = ex.apply(make_op("map_values", "s", mapping={"zz": "y"}), approved_by="user")
    assert e.rows_affected == 0 and e.note == "no rows changed"
    r = ex.reject(make_op("drop_column", "n"), approved_by="farrel")
    assert r.action == "reject" and r.rows_affected == 0 and "n" in ex.current.columns
    assert [x.seq for x in ex.audit_log] == [1, 2]


def test_audit_log_accessor_returns_a_copy():
    ex = Executor(frame(5))
    ex.apply(make_op("strip_whitespace", "s"), approved_by="user")
    ex.audit_log.clear()
    assert len(ex.audit_log) == 1


# ---- replay determinism (foundation of recipe.json) ---------------------------------------------------------------------------
def test_replaying_applied_operations_reproduces_the_file_byte_for_byte():
    df = messy_frame()
    prof = profile_dataframe(df)
    ex = Executor(df)
    for p in rules_only_proposals(detect_all(df, prof), prof):
        ex.apply(p.operation, confirm_large_drop=True, approved_by="user")
    recipe = json.dumps([o.model_dump() for o in ex.applied_operations()])

    replay = Executor(df.copy())
    for raw in json.loads(recipe):
        replay.apply(Operation.model_validate(raw), confirm_large_drop=True, approved_by="user")  # no LLM, same code path
    a = ex.current.to_csv(index=False).encode()
    b = replay.current.to_csv(index=False).encode()
    assert a == b and len(a) > 1000
