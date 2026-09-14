"""
Functional proof for analysis_service.py.

WHAT THIS CAN TEST HERE
    _json_safe_scalar, _dataframe_to_records, _shape_result,
    _compute_confidence, _is_plotly_fig, _is_matplotlib_fig — none of
    these need query_engine.py or llm_provider.py, so they're fully
    testable without streamlit installed.
    Also: run_analysis()'s CSV-parse-failure path, which returns before
    ever reaching the lazy query_engine import.

WHAT THIS CANNOT TEST HERE (and why)
    The real end-to-end pipeline — a valid CSV + a real query going all
    the way through detect_shortcut/build_analysis_prompt/llm_with_fallback
    /run_sandboxed — needs query_engine.py, which needs llm_provider.py,
    which needs streamlit (for st.cache_data). streamlit isn't installed
    in this sandbox and there's no network here to install it. This is
    the same limitation test_isolation.py hit earlier in this project —
    RUN THIS ON YOUR MACHINE, where streamlit/fastapi/pydantic already
    exist, to get real coverage of the shortcut-query path. See the
    bottom of this file for exactly what to check.

Run: python3 test_analysis_service.py
"""

import io
import numpy as np
import pandas as pd

from analysis_service import (
    _json_safe_scalar,
    _dataframe_to_records,
    _shape_result,
    _compute_confidence,
    _is_plotly_fig,
    _is_matplotlib_fig,
    run_analysis,
)


def main():
    print("=" * 70)
    print("PART 1 — JSON-safety conversions")
    print("=" * 70)

    cases = [
        ("numpy int64", np.int64(42), 42),
        ("numpy float64", np.float64(3.14), 3.14),
        ("numpy bool_", np.bool_(True), True),
        ("plain python int (passthrough)", 7, 7),
        ("NaN -> None", float("nan"), None),
    ]
    for label, value, expected in cases:
        result = _json_safe_scalar(value)
        status = "OK" if result == expected or (result is None and expected is None) else "!!! FAILED !!!"
        print(f"[{status}] {label}: {value!r} -> {result!r}")

    print("\n" + "=" * 70)
    print("PART 2 — DataFrame -> JSON-safe records")
    print("=" * 70)

    df = pd.DataFrame({
        "name": ["Alice", "Bob"],
        "sales": [np.int64(1200), np.int64(2500)],
        "rating": [np.float64(4.5), None],
    })
    records = _dataframe_to_records(df)
    print(f"records: {records}")
    assert records[0]["sales"] == 1200 and isinstance(records[0]["sales"], int), "int64 not converted"
    assert records[1]["rating"] is None, "null not converted to None"
    print("[OK] DataFrame converts cleanly to JSON-safe records")

    print("\n" + "=" * 70)
    print("PART 3 — result shaping by type")
    print("=" * 70)

    shaped_scalar = _shape_result(np.int64(99))
    print(f"scalar -> {shaped_scalar}")
    assert shaped_scalar["resultType"] == "scalar" and shaped_scalar["resultData"] == 99
    print("[OK] scalar shaping")

    shaped_table = _shape_result(df)
    print(f"table -> resultType={shaped_table['resultType']}, rows={len(shaped_table['resultData'])}")
    assert shaped_table["resultType"] == "table"
    print("[OK] DataFrame shaping")

    shaped_series = _shape_result(df["sales"])
    print(f"series -> resultType={shaped_series['resultType']}")
    assert shaped_series["resultType"] == "table"
    print("[OK] Series shaping (converted to table)")


    class FakeMplFigure:
        """Stands in for a real matplotlib Figure without needing matplotlib
        installed for this specific test — only needs .savefig and .gca to
        exist, matching what _is_matplotlib_fig checks for."""
        def savefig(self, *a, **k): pass
        def gca(self): pass


    shaped_mpl = _shape_result(FakeMplFigure())
    print(f"matplotlib figure -> {shaped_mpl}")
    assert shaped_mpl["resultType"] == "error", "matplotlib figures should return a clear error, not crash"
    print("[OK] matplotlib figure correctly returns an explicit 'not supported yet' error")

    print("\n" + "=" * 70)
    print("PART 4 — confidence scoring")
    print("=" * 70)

    conf_high = _compute_confidence(df, success=True, shortcut_used=True)
    conf_low = _compute_confidence(None, success=False, shortcut_used=False)
    print(f"DataFrame + success + shortcut -> {conf_high} (expect HIGH)")
    print(f"None + failure + no shortcut -> {conf_low} (expect LOW)")
    assert conf_high == "HIGH" and conf_low == "LOW"
    print("[OK] confidence scoring matches expected tiers")

    print("\n" + "=" * 70)
    print("PART 5 — run_analysis() error path (no streamlit needed)")
    print("=" * 70)

    result = run_analysis(b"this is not a csv file at all {{{", "test.csv", "any query", "session-1")
    print(f"garbage input -> {result}")
    assert result["success"] is False and result["resultType"] == "error"
    print("[OK] malformed CSV input fails cleanly, before ever touching query_engine")

    print("\n" + "=" * 70)
    print("PART 6 — the real pipeline (SKIPPED here, run on your machine)")
    print("=" * 70)

    valid_csv = b"name,sales,category\nAlice,1200,Electronics\nBob,2500,Clothing\n"
    result = run_analysis(valid_csv, "sales.csv", "how many rows", "session-1")

    if not result["success"] and "streamlit" in str(result.get("error", "")).lower():
        print(f"[SKIPPED] {result['error']}")
        print("          Expected in this sandbox — streamlit isn't installed here.")
        print("          On your machine, this should print [OK] with a real result.")
    elif result["success"] and result["resultType"] == "scalar" and result["resultData"] == 2:
        print(f"[OK] shortcut-layer query worked end-to-end -> {result}")
    else:
        print(f"[!!! CHECK THIS !!!] unexpected result: {result}")

    print("\n" + "=" * 70)
    print("Done.")
    print("If you're running this on your machine (with streamlit/fastapi/")
    print("pydantic installed), Part 6 should show [OK], not [SKIPPED].")



if __name__ == "__main__":
    main()
