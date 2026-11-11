"""Shared test helpers (no LLM, no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.schemas import Operation


def make_op(op: str, column: str | None = None, **params) -> Operation:
    return Operation(op=op, column=column, params=params)


def messy_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A deterministic frame containing every defect the detectors look for."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "id": range(1, n + 1),
        "age": rng.integers(18, 70, n).astype(float),
        "income": rng.normal(50000, 8000, n),
        "country": rng.choice(["USA", "usa", "U.S.A.", "India", "india", "Germany"], n),
        "city": rng.choice(["  Paris", "London ", "Berlin"], n),
        "joined": pd.Series(pd.date_range("2020-01-01", periods=n)).dt.strftime("%d/%m/%Y"),
        "score": rng.integers(0, 100, n).astype(str),
        "const": "x",
    })
    df.loc[3:9, "age"] = np.nan
    df.loc[10, "income"] = 9e6
    df.loc[20:25, "city"] = None
    return pd.concat([df, df.iloc[:3]], ignore_index=True)
