import numpy as np
import pandas as pd
import pytest

from core.profiler import profile_dataframe
from tests.helpers import messy_frame


def test_kinds_and_null_counting_with_blanks():
    df = pd.DataFrame({
        "n": [1.0, 2.0, None, 4.0],
        "b": [True, False, True, True],
        "d": pd.to_datetime(["2024-01-01"] * 4),
        "s": ["a", "  ", None, "a"],
        "e": [None, None, None, None],
    })
    p = profile_dataframe(df)
    kinds = {c.name: c.kind for c in p.columns}
    assert kinds == {"n": "numeric", "b": "bool", "d": "datetime", "s": "categorical", "e": "empty"}
    s = p.column("s")
    assert s.null_count == 2 and s.null_pct == 50.0 and s.n_unique == 1
    assert p.column("n").null_count == 1 and p.column("e").null_pct == 100.0


def test_numeric_stats_and_skew():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]
    c = profile_dataframe(pd.DataFrame({"n": x})).column("n")
    assert c.median == 5.5 and c.max == 1000 and c.min == 1 and c.mean == pytest.approx(104.5)
    assert c.skew is not None and c.skew > 2


def test_samples_are_capped_compact_and_most_frequent_first():
    df = pd.DataFrame({"c": ["common"] * 5 + [f"value-{i}-" + "x" * 80 for i in range(30)]})
    c = profile_dataframe(df).column("c")
    assert len(c.sample_values) == 8 and c.sample_values[0] == "'common'"
    assert all(len(v) <= 40 for v in c.sample_values)


def test_kind_categorical_vs_text_vs_id():
    n = 100
    df = pd.DataFrame({
        "cat": ["a", "b"] * (n // 2),
        "free": [f"unique sentence number {i}" for i in range(n)],
        "uid": range(n),
    })
    p = profile_dataframe(df)
    assert p.column("cat").kind == "categorical"
    assert p.column("free").kind == "text" and p.column("free").looks_like_id
    assert p.column("uid").looks_like_id and not p.column("cat").looks_like_id


def test_duplicate_rows_counted():
    df = pd.DataFrame({"a": [1, 1, 1, 2], "b": ["x", "x", "x", "y"]})
    assert profile_dataframe(df).duplicate_rows == 2


def test_signature_is_stable_and_sensitive():
    df = messy_frame()
    a, b = profile_dataframe(df), profile_dataframe(df.copy())
    assert a.signature() == b.signature()
    changed = df.copy()
    changed.loc[0, "age"] = np.nan
    assert profile_dataframe(changed).signature() != a.signature()
    assert a.column("age").signature() == b.column("age").signature()


def test_rejects_duplicate_or_non_string_column_names():
    with pytest.raises(ValueError, match="Duplicate"):
        profile_dataframe(pd.DataFrame([[1, 2]], columns=["a", "a"]))
    with pytest.raises(ValueError, match="strings"):
        profile_dataframe(pd.DataFrame([[1, 2]], columns=[0, 1]))


def test_empty_frame_does_not_crash():
    p = profile_dataframe(pd.DataFrame({"a": pd.Series([], dtype=float)}))
    assert p.n_rows == 0 and p.column("a").kind == "empty"
