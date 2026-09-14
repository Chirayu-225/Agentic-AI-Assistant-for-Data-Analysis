"""
Functional proof for table_extraction.py.

Generates two synthetic PDFs with reportlab (already a project
dependency) rather than requiring a fixture file: one with a real
table, one with prose only — proving both that a real table is
correctly extracted AND that the noise filter doesn't false-positive
on ordinary paragraph text.

Also proves the actual end-to-end claim this feature exists for: an
extracted table decodes and flows through file_reader.py + cleaning.py
— the SAME pipeline a regular CSV upload goes through — not a
separate, unverified code path.

Run: python3 test_table_extraction.py
"""

import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from table_extraction import extract_tables_from_pdf
from file_reader import read_tabular_file
from cleaning import rule_based_clean

print("=" * 70)
print("PART 1 — a PDF with a real table")
print("=" * 70)

buf = io.BytesIO()
doc = SimpleDocTemplate(buf)
styles = getSampleStyleSheet()
story = [Paragraph("Quarterly Revenue by Region", styles["Heading2"])]
data = [
    ["Region", "Q1", "Q2", "Q3", "Q4"],
    ["North America", "12000", "13500", "14200", "15800"],
    ["Europe", "8000", "8200", "7900", "8600"],
    ["MEA", "3000", "3400", "3900", "4600"],
]
table = Table(data)
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
    ("GRID", (0, 0), (-1, -1), 1, colors.black),
]))
story.append(table)
doc.build(story)
pdf_with_table = buf.getvalue()

results = extract_tables_from_pdf(pdf_with_table, "test.pdf")
print(f"\n{len(results)} table(s) found")
assert len(results) == 1, f"expected exactly 1 table, got {len(results)}"
t = results[0]
assert t["row_count"] == 3 and t["col_count"] == 5
print(f"  {t['title']}: {t['row_count']} rows x {t['col_count']} cols")
print("[OK] real table correctly detected and extracted")

print("\n" + "=" * 70)
print("PART 2 — a prose-only PDF (noise filter should find nothing)")
print("=" * 70)

buf2 = io.BytesIO()
doc2 = SimpleDocTemplate(buf2)
story2 = [Paragraph(
    "This is a plain narrative document with no tables at all, just "
    "prose describing quarterly performance without any grid structure.",
    styles["Normal"],
)]
doc2.build(story2)
pdf_no_table = buf2.getvalue()

results2 = extract_tables_from_pdf(pdf_no_table, "prose.pdf")
print(f"\n{len(results2)} table(s) found (should be 0)")
assert results2 == [], "noise filter should not false-positive on prose text"
print("[OK] noise filter correctly finds nothing in prose-only content")

print("\n" + "=" * 70)
print("PART 3 — the whole point: extracted table flows through the SAME")
print("pipeline a regular CSV upload goes through, unmodified")
print("=" * 70)

csv_bytes = t["csv_bytes"]
df = read_tabular_file(csv_bytes, "extracted_table.csv")
print(f"\nparsed shape: {df.shape}")
assert df.shape == (3, 5)

df_clean, log = rule_based_clean(df)
print("cleaning log:", log)
print(df_clean.dtypes.to_dict())
assert df_clean["q1"].dtype.kind in "if", "revenue columns should be real numeric dtype, not strings"
print("[OK] extracted table is correctly typed real data — same guarantees as a CSV upload")

print("\n" + "=" * 70)
print("Done.")
