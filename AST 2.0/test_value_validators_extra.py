"""Two extra validator behaviours: fractions among whole percentages, and ship-before-order date pairs."""
import datetime as dt
import numpy as np
import pandas as pd
import value_validators as vv

TODAY = dt.date(2026, 11, 11)
FAILS = []
def check(label, cond, detail=""):
    print(("[OK] " if cond else "[FAIL] ") + label + ("" if cond else f"  {detail}"))
    if not cond: FAILS.append(label)

n = 200
disc = pd.DataFrame({"discount_pct": [0.0, 5.0, 10.0, 15.0, 20.0] * 40})
d1 = disc.copy(); d1.loc[[6, 12, 33], "discount_pct"] = [0.05, 0.1, 0.15]      # 5, 10, 15 written as fractions
out, rep = vv.validate_values(d1, today=TODAY)
check("whole-percent column: fractions whose x100 already occurs are multiplied by 100", out.loc[[6, 12, 33], "discount_pct"].tolist() == [5.0, 10.0, 15.0], str(out["discount_pct"][[6, 12, 33]].tolist()))
check("whole-percent column: zeros and whole values untouched", (out["discount_pct"].drop([6, 12, 33]) == d1["discount_pct"].drop([6, 12, 33])).all())
check("whole-percent column: the change is reported", [(e["rule"], e["count"]) for e in rep["entries"]] == [("percent_scale", 3)], str(rep["entries"]))

d2 = disc.copy(); d2.loc[[6, 12, 33], "discount_pct"] = [0.5, 0.75, 2.5]       # genuine small percentages: x100 is not a known value
check("genuine small percentages are left alone", vv.validate_values(d2, today=TODAY)[1]["entries"] == [])

rng = np.random.default_rng(0)
od = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
dates = pd.DataFrame({"order_date": od, "ship_date": od + pd.to_timedelta(rng.integers(1, 9, n), unit="D")})
bad = dates.copy(); bad.loc[[4, 9, 50], "ship_date"] = bad.loc[[4, 9, 50], "order_date"] - pd.Timedelta(days=5)
out, rep = vv.validate_values(bad, today=TODAY)
check("ship_date before order_date is blanked and reported", out["ship_date"].isna().sum() == 3 and out.loc[[4, 9, 50], "ship_date"].isna().all() and rep["entries"][0]["rule"] == "end_before_start", str(rep["entries"]))
check("order_date is never touched", out["order_date"].equals(bad["order_date"]))
rev = dates.copy(); rev["ship_date"] = rev["order_date"] - pd.Timedelta(days=3)    # reversed columns: a majority violates, so abstain
check("majority violating: abstains (probably not a start/end pair)", vv.validate_values(rev, today=TODAY)[1]["entries"] == [])

# future dates: a calendar tail is not typos
cal = pd.DataFrame({"request_date": pd.Timestamp("2026-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 365, 3000)), unit="D")})
check("a calendar that simply runs past today (dense future days) is left alone", vv.validate_values(cal, today=dt.date(2026, 11, 11))[1]["entries"] == [])
typo = pd.DataFrame({"order_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, 300), unit="D")})
typo.loc[[3, 40], "order_date"] = [pd.Timestamp("2031-03-14"), pd.Timestamp("2029-07-01")]
out, rep = vv.validate_values(typo, today=TODAY)
check("a few scattered impossible dates are still blanked", out["order_date"].isna().sum() == 2 and rep["entries"][0]["rule"] == "future_date", str(rep["entries"]))
print("\nDONE — all checks passed." if not FAILS else f"\n{len(FAILS)} FAILED")
raise SystemExit(1 if FAILS else 0)
