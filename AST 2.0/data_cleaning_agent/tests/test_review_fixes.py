"""Provenance rule for map_values, approval policy, bounded executor memory."""
import pandas as pd
import pytest

from core.executor import Executor
from core.operations import OperationError
from core.policy import PolicyError, restriction
from core.preview import preview
from tests.helpers import make_op


def df():
    return pd.DataFrame({"kind": ["Plane", "plane", "Car", "Car", "Boat"], "n": [1.0, 2.0, 3.0, None, 5.0]})


def mv(source, mapping):
    op = make_op("map_values", "kind", mapping=mapping)
    return op.model_copy(update={"source": source})


def test_llm_mapping_to_invented_value_is_rejected_everywhere():
    op = mv("llm", {"Plane": "Aircraft"})
    pv = preview(df(), op)
    assert not pv.ok and "not found" in pv.error
    with pytest.raises(OperationError):
        Executor(df()).apply(op, approved_by="user")


def test_llm_mapping_onto_existing_value_ok():
    ex = Executor(df())
    ex.apply(mv("llm", {"plane": "Plane"}), approved_by="user")
    assert list(ex.current["kind"][:2]) == ["Plane", "Plane"]


def test_user_typed_value_and_trusted_replay_are_exempt():
    Executor(df()).apply(mv("user", {"Plane": "Aircraft"}), approved_by="user")
    Executor(df(), strict=False).apply(mv("llm", {"Plane": "Aircraft"}), approved_by="user")


@pytest.mark.parametrize("op", [
    make_op("impute", "n", strategy="constant", value=0),
    make_op("impute", "n", strategy="median"),
    make_op("drop_column", "n"),
    make_op("clip_outliers", "n", method="iqr"),
    make_op("drop_duplicates"),
    mv("llm", {"plane": "Plane"}),
    mv("rule", {"plane": "Plane"}),   # every value mapping needs a person, whoever proposed it
    mv("user", {"plane": "Plane"}),
])
@pytest.mark.parametrize("who", [None, "auto", "rule", "llm", ""])  # None = nobody said (the default)
def test_restricted_ops_need_a_person(op, who):
    ex = Executor(df())
    assert restriction(op)
    with pytest.raises(PolicyError):
        ex.apply(op, approved_by=who)
    assert ex.version == 0 and ex.audit_log == []


def test_policy_tag_and_source_recorded():
    ex = Executor(df())
    e = ex.apply(make_op("impute", "n", strategy="constant", value=0), approved_by="user")
    assert e.policy == "impute-constant" and e.approved_by == "user" and e.source == "rule"
    assert ex.audit_records()[0]["policy"] == "impute-constant"
    e2 = ex.apply(make_op("strip_whitespace", "kind"), approved_by="auto")  # harmless ops may be automatic
    assert e2.policy is None


def test_memory_bounded_and_undo_all_the_way_back():
    big = pd.DataFrame({"t": ["  x "] * 50, "v": range(50)})
    ex = Executor(big, keep_snapshots=2)
    ops = [make_op("strip_whitespace", "t"), make_op("normalize_case", "t", mode="upper"),
           make_op("impute", "v", strategy="median"), make_op("clip_outliers", "v", method="iqr"),
           make_op("map_values", "t", mapping={"X": "X"}), make_op("drop_duplicates")]
    states = []
    for op in ops:
        states.append(ex.current.copy())
        ex.apply(op, approved_by="user")
    resident = sum(f is not None for f in ex._snapshots)
    assert resident <= 1 + 2 + 1  # original + keep + newest
    while ex.can_undo:
        ex.undo()
        pd.testing.assert_frame_equal(ex.current, states.pop())
    assert ex.version == 0


def test_byte_budget_evicts():
    ex = Executor(pd.DataFrame({"v": range(1000)}), keep_snapshots=50, max_resident_bytes=1)
    for _ in range(4):
        ex.apply(make_op("clip_outliers", "v", method="iqr"), approved_by="user")
    assert sum(f is not None for f in ex._snapshots) == 2  # original + current only
    ex.undo(); ex.undo()
    assert ex.version == 2
