"""
Displayed numbers carry no floating-point noise.

Run from the "AST 2.0" folder:
    python -m tests.analysis.test_display_rounding

A sum of money came back as 4853917.909999996 (the true value is
4853917.91): binary floats cannot hold most decimals exactly and every
addition leaves a tail. Numbers are rounded to 15 significant digits when
they are shaped for the user. That is the precision a float can promise,
so no real digit is lost. The verification step compares the unrounded
values before this runs, so it is not weakened.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FAIL: list[str] = []


def check(label: str, condition: bool, detail: str = ""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(label)


def main():
    from analysis.analysis_service import _json_safe_scalar, _shape_result

    check("the tail from the eval is removed", _json_safe_scalar(4853917.909999996) == 4853917.91)
    check("a numpy float too", _json_safe_scalar(np.float64(4005.0000000000005)) == 4005.0)
    check("the classic 0.1 + 0.2", _json_safe_scalar(0.1 + 0.2) == 0.3)
    check("a real repeating decimal keeps its digits", _json_safe_scalar(1 / 3) == float(f"{1/3:.15g}")
          and str(_json_safe_scalar(1 / 3)).startswith("0.33333333333333"))
    check("integers are untouched", _json_safe_scalar(123456789012345678) == 123456789012345678)
    check("a numpy integer stays an int", isinstance(_json_safe_scalar(np.int64(7)), int))
    check("cents survive on a large amount", _json_safe_scalar(123456789012.34) == 123456789012.34)
    check("zero and negatives", _json_safe_scalar(0.0) == 0.0 and _json_safe_scalar(-2.5000000000000004) == -2.5)
    check("infinity is not turned into something else", math.isinf(_json_safe_scalar(float("inf"))))
    check("text is untouched", _json_safe_scalar("1.0000000000000002") == "1.0000000000000002")

    out = _shape_result(4853917.909999996)
    check("a scalar answer is rounded", out["resultData"] == 4853917.91, repr(out))
    df = pd.DataFrame({"region": ["N", "S"], "total": [4853917.909999996, 4173195.789999999], "n": [3, 4]})
    out = _shape_result(df)
    check("table cells are rounded", [r["total"] for r in out["resultData"]] == [4853917.91, 4173195.79],
          repr(out["resultData"]))
    check("integer columns stay integers", [r["n"] for r in out["resultData"]] == [3, 4]
          and all(isinstance(r["n"], int) for r in out["resultData"]))

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE - {len(FAIL)} FAILURE(S): {FAIL}")
        raise SystemExit(1)
    print("DONE - all checks passed.")


if __name__ == "__main__":
    main()
