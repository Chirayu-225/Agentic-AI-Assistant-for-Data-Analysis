"""
test_pii_detection.py — the expanded PII layer: Indian identifiers, checksum
validation, false-positive resistance, column-name withholding, the
DataFrame scan behind /profile's piiReport, the optional NER hook, and a
check that the real prompt builders actually apply all of it.

Run: python3 test_pii_detection.py
"""

import os
import sys

import pandas as pd

os.environ.setdefault("AUDIT_LOG_DISABLED", "1")

import pii_redaction as pii

FAIL = []


def check(label, condition, detail=""):
    print(f"[{'OK' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(label)


print("=" * 70)
print("PART 1 — checksum validators against independent known vectors")
print("=" * 70)
# 236 -> check digit 3 is the standard worked Verhoeff example.
check("Verhoeff accepts the textbook vector 2363", pii.verhoeff_valid("2363"))
check("Verhoeff rejects 2364", not pii.verhoeff_valid("2364"))
base = "23456789012"
aadhaar = base + str(pii.verhoeff_check_digit(base))
check("generated Aadhaar-shaped number validates", pii.verhoeff_valid(aadhaar), aadhaar)
check("Luhn accepts the standard test card 4111111111111111", pii.luhn_valid("4111111111111111"))
check("Luhn rejects 4111111111111112", not pii.luhn_valid("4111111111111112"))

print("\n" + "=" * 70)
print("PART 2 — each identifier type is redacted under its own label")
print("=" * 70)
spaced = f"{aadhaar[:4]} {aadhaar[4:8]} {aadhaar[8:]}"
cases = [
    ("email", "mail attacker@evil.com now", "attacker@evil.com", "EMAIL"),
    ("UPI id", "pay john.doe@okhdfcbank today", "john.doe@okhdfcbank", "UPI_ID"),
    ("card (grouped)", "card 4111 1111 1111 1111 ok", "4111 1111 1111 1111", "CREDIT_CARD"),
    ("card (bare, Luhn-valid)", "card 4111111111111111 ok", "4111111111111111", "CREDIT_CARD"),
    ("Aadhaar (bare)", f"id {aadhaar} end", aadhaar, "AADHAAR"),
    ("Aadhaar (grouped)", f"id {spaced} end", spaced, "AADHAAR"),
    ("US SSN", "ssn 123-45-6789", "123-45-6789", "SSN"),
    ("PAN", "PAN ABCPE1234F on file", "ABCPE1234F", "PAN"),
    ("Indian mobile (bare)", "call 9876543210 now", "9876543210", "INDIAN_MOBILE"),
    ("Indian mobile (+91 5-5)", "call +91 98765 43210 now", "98765 43210", "INDIAN_MOBILE"),
    ("Indian mobile (+91-)", "call +91-9876543210 now", "9876543210", "INDIAN_MOBILE"),
    ("separator phone", "call 555-867-5309 now", "555-867-5309", "PHONE"),
]
for label, text, raw, expect_label in cases:
    out, counts = pii.redact_pii(text)
    check(f"redacts {label} as {expect_label}", raw not in out and counts.get(expect_label) == 1, f"{out!r} {counts}")

print("\n" + "=" * 70)
print("PART 3 — things that look numeric but must NOT be redacted")
print("=" * 70)
keep = [
    ("business sentence", "Revenue grew 12% in Q3 driven by the Enterprise segment."),
    ("statistics line with big floats", "mean=7234567890.5 max=9876543210.0 min=1696118400000.0"),
    ("13-digit timestamp (fails Luhn, no separators)", "created 1696118400000 by system"),
    ("Aadhaar-shaped but fails Verhoeff", "ref 234567890123 pending"),
    ("order total", "order id 5512 total 12345678"),
    ("plain year and quantity", "In 2024 we sold 1500 units"),
]
for label, text in keep:
    out, counts = pii.redact_pii(text)
    check(f"unchanged: {label}", out == text and not counts, f"{out!r} {counts}")

print("\n" + "=" * 70)
print("PART 4 — column-name classification")
print("=" * 70)
expect = {
    "first_name": "personal", "customerName": "personal", "Employee Name": "personal",
    "billing_address": "personal", "user_email": "personal", "Phone": "personal",
    "mobile_number": "personal", "dob": "personal", "aadhaar": "personal", "pan_number": "personal",
    "name": "review",
    "product_name": None, "city": None, "email_verified": None, "phone_model": None,
    "revenue": None, "region": None, "company_name": None,
}
for col, want in expect.items():
    got = pii.classify_column_name(col)
    check(f"classify {col!r} -> {want}", got == want, f"got {got}")

print("\n" + "=" * 70)
print("PART 5 — personal columns are withheld from profiles, and the prompts")
print("=" * 70)
df = pd.DataFrame({
    "customer_name": ["Asha Rao", "Vikram Shah", "Neha Iyer", "Asha Rao"],
    "email": ["asha@x.com", "vikram@y.org", "neha@z.in", "asha@x.com"],
    "phone": [9876543210, 9123456780, 9988776655, 9876543210],
    "city": ["Pune", "Delhi", "Pune", "Pune"],
    "product_name": ["Laptop", "Phone", "Laptop", "Laptop"],
    "amount": [120.5, 300.0, 99.9, 120.5],
})

from profiler import (build_dataset_profile, format_profile_for_llm,
                      build_defect_profile, format_defect_profile_for_llm)

for fmt_name, text in (
    ("dataset profile", format_profile_for_llm(build_dataset_profile(df))),
    ("defect profile", format_defect_profile_for_llm(build_defect_profile(df))),
):
    for secret in ("Asha Rao", "Vikram Shah", "asha@x.com", "9876543210", "9123456780"):
        check(f"{fmt_name} never contains {secret!r}", secret not in text)
    check(f"{fmt_name} still shows non-personal values (city)", "Pune" in text)
    check(f"{fmt_name} still shows non-personal product names", "Laptop" in text)
    check(f"{fmt_name} states values were withheld", pii.WITHHELD in text)

masked = pii.mask_personal_columns(df.head(2))
check("mask_personal_columns hides name/email/phone", (masked["customer_name"] == pii.WITHHELD).all()
      and (masked["email"] == pii.WITHHELD).all() and (masked["phone"] == pii.WITHHELD).all())
check("mask_personal_columns keeps city", list(masked["city"]) == ["Pune", "Delhi"])
check("mask_personal_columns does not modify the original", df["customer_name"].iloc[0] == "Asha Rao")

os.environ["PII_WITHHOLD_PERSONAL_COLUMNS"] = "0"
check("withholding can be switched off by env var", pii.personal_columns(df.columns) == [])
del os.environ["PII_WITHHOLD_PERSONAL_COLUMNS"]

print("\n" + "=" * 70)
print("PART 6 — scan_dataframe (the /profile piiReport)")
print("=" * 70)
report = pii.scan_dataframe(df)
flagged = {f["column"]: f for f in report["flaggedColumns"]}
check("report risk is high", report["risk"] == "high", report["risk"])
for col in ("customer_name", "email", "phone"):
    check(f"{col} is flagged", col in flagged)
check("email flagged by value pattern", "EMAIL" in flagged["email"]["categories"])
check("phone flagged by value pattern", "INDIAN_MOBILE" in flagged["phone"]["categories"])
check("customer_name flagged by NAME and withheld", flagged["customer_name"]["withheldFromLlm"])
check("city / product_name / amount not flagged",
      not any(c in flagged for c in ("city", "product_name", "amount")))
clean_df = pd.DataFrame({"region": ["N", "S"], "sales": [10, 20]})
check("clean table -> risk none, no flagged columns",
      pii.scan_dataframe(clean_df)["risk"] == "none" and not pii.scan_dataframe(clean_df)["flaggedColumns"])
check("scan does not modify the DataFrame", df["email"].iloc[0] == "asha@x.com")

print("\n" + "=" * 70)
print("PART 7 — optional NER hook (fake backend; the spaCy adapter itself is not tested here)")
print("=" * 70)
def fake_ner(text):
    spans, i = [], text.find("Priya Menon")
    if i >= 0:
        spans.append((i, i + len("Priya Menon")))
    return spans

pii.set_ner_backend(fake_ner)
out, counts = pii.redact_pii("Complaint raised by Priya Menon about delivery")
check("injected NER backend redacts a person name", "Priya Menon" not in out and counts.get("NAME") == 1, out)
pii.set_ner_backend(lambda t: (_ for _ in ()).throw(RuntimeError("model crashed")))
out, counts = pii.redact_pii("Complaint raised by Priya Menon")
check("a crashing NER backend degrades to pattern-only, not an exception", out == "Complaint raised by Priya Menon")
pii.set_ner_backend(None)

print("\n" + "=" * 70)
print("PART 8 — the prompt builders that embed real file content")
print("=" * 70)
import analysis_service  # noqa: F401  (import check only)
from multi_table_service import _build_schema_text
text = _build_schema_text({"customers": df})
for secret in ("Asha Rao", "asha@x.com", "9876543210"):
    check(f"multi-table schema prompt never contains {secret!r}", secret not in text)
check("multi-table schema prompt is fenced as untrusted data", "BEGIN UNTRUSTED DATA" in text)

print("\n" + "=" * 70)
if FAIL:
    print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("DONE — all checks passed.")
