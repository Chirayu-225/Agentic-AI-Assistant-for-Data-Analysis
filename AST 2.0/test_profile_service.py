"""
Functional proof for profile_service.py.

Unlike analysis_service.py / rag_service.py, this module has zero
streamlit dependency, so this test runs COMPLETELY here — no [SKIPPED]
sections, no "run this on your machine" caveat needed for the core logic.

Run: python3 test_profile_service.py
"""

from profile_service import get_profile, get_cleaning_report

SAMPLE_CSV = b"""name,sales,category,signup_date,discount
Alice,$1200,Electronics,2024-01-15,10%
Bob,$2500,Clothing,2024-02-20,N/A
Charlie,,Electronics,2024-01-30,15%
Dave,$1800,Clothing,invalid_date,5%
"""

print("=" * 70)
print("PART 1 — raw profile (pre-cleaning)")
print("=" * 70)

result = get_profile(SAMPLE_CSV, "test.csv")
print(f"\n{result}")

assert result["success"] is True
assert result["totalRows"] == 4
assert result["totalColumns"] == 5
assert result["nullValues"] >= 1  # Charlie's missing sales
assert len(result["schema"]) == 5
assert any(col["column"] == "sales" for col in result["schema"])
print("\n[OK] raw profile: correct row/column counts, schema present")

# sales column should be raw (unparsed currency), still str-ish before cleaning
sales_schema = next(c for c in result["schema"] if c["column"] == "sales")
print(f"sales column dtype before cleaning: {sales_schema['dtype']}")
assert sales_schema["dtype"] == "str", "raw profile should show unparsed currency as str, not numeric"
print("[OK] raw profile correctly shows uncleaned currency as 'str', not a number")

print("\n" + "=" * 70)
print("PART 2 — cleaning report (before/after)")
print("=" * 70)

result = get_cleaning_report(SAMPLE_CSV, "test.csv")
print(f"\nlog: {result.get('log')}")
print(f"before: {result.get('before')}")
print(f"after: {result.get('after')}")

assert result["success"] is True
assert len(result["log"]) > 0, "cleaning should report at least one action taken"
print("\n[OK] cleaning report includes a non-empty action log")

sales_schema_after = next(c for c in result["schema"] if c["column"] == "sales")
print(f"sales column dtype after cleaning: {sales_schema_after['dtype']}")
assert sales_schema_after["dtype"] in ("int64", "float64"), "currency should be parsed to numeric after cleaning"
print("[OK] currency ('$1200' etc.) correctly parsed to numeric after cleaning")

assert result["after"]["nullValues"] <= result["before"]["nullValues"] or result["after"]["nullValues"] == 0, \
    "cleaning should fill nulls, not increase them"
print(f"[OK] null count did not increase: before={result['before']['nullValues']} after={result['after']['nullValues']}")

print("\n" + "=" * 70)
print("PART 3 — error handling on genuinely broken input")
print("=" * 70)

result = get_profile(b"", "empty.csv")
print(f"\nempty file -> {result}")
assert result["success"] is False
print("[OK] empty file fails cleanly, no crash")

print("\n" + "=" * 70)
print("Done. All parts ran for real in this sandbox — no [SKIPPED] sections.")
