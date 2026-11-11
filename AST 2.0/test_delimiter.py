"""CSV separator detection: semicolon/tab/pipe files read as tables, comma files read exactly as before."""
import io
import pandas as pd
import file_reader as fr
FAILS = []
def check(label, cond, detail=""):
    print(("[OK] " if cond else "[FAIL] ") + label + ("" if cond else f"  {detail}"))
    if not cond: FAILS.append(label)

rows = [["ORD1", "C1", "[1, 2, 3]", "10.5"], ["ORD2", "C2", "[4, 5, 6]", "7"], ["ORD3", "C3", "[7, 8, 9]", "3.25"]] * 5
def make(sep, text_fields_with_commas=True):
    lines = [sep.join(["order_id", "customer_id", "desc", "price"])] + [sep.join(r) for r in rows]
    return ("\r\n".join(lines) + "\r\n").encode()

check("comma file: comma", fr.detect_delimiter(b"a,b,c\n1,2,3\n4,5,6\n") == ",")
check("semicolon file whose values contain commas: semicolon", fr.detect_delimiter(make(";")) == ";")
check("tab file: tab", fr.detect_delimiter(make("\t")) == "\t")
check("pipe file: pipe", fr.detect_delimiter(make("|")) == "|")
check("one-column file: comma (no change)", fr.detect_delimiter(b"name\nann\nbob\n") == ",")
check("empty / tiny input: comma", fr.detect_delimiter(b"") == "," and fr.detect_delimiter(b"x") == ",")
check("semicolons only inside quoted comma fields: comma", fr.detect_delimiter(b'a,b\n"x;y",1\n"p;q",2\n') == ",")
check("binary junk never raises", fr.detect_delimiter(bytes(range(256)) * 20) in (",", ";", "\t", "|"))

for pol in ("legacy", "safe"):
    df = fr._read_tabular_file_uncached(make(";"), "orders.csv", na_policy=pol)
    check(f"{pol}: a semicolon file reads as 4 columns, 15 rows", df.shape == (15, 4) and list(df.columns)[0] == "order_id", str(df.shape))
    check(f"{pol}: values containing commas stay whole", df["desc"].iloc[0] == "[1, 2, 3]")
df = fr._read_tabular_file_uncached(make(";"), "orders.csv", na_policy="safe")
check("safe: the report says how the file was read", any("semicolon" in n for n in df.attrs.get("reader_notes", [])))
dfc = fr._read_tabular_file_uncached(b"a,b\n1,2\n3,4\n", "x.csv", na_policy="safe")
check("a comma file gets no separator note", not dfc.attrs.get("reader_notes"))
lat = make(";").replace(b"ORD1", "ÖRD1".encode("latin1"), 1)
check("latin1 fallback keeps the separator", fr._read_tabular_file_uncached(lat, "o.csv", na_policy="safe").shape == (15, 4))
# real "NA" in a semicolon file is still kept by the safe policy
na = b"id;territory;x\n1;NA;5\n2;NA;6\n3;EMEA;7\n" * 1
na = b"id;territory;x\n" + b"".join(f"{i};{'NA' if i % 2 else 'EMEA'};{i}\n".encode() for i in range(40))
check("safe: a real NA label survives in a semicolon file", (fr._read_tabular_file_uncached(na, "t.csv", na_policy="safe")["territory"] == "NA").sum() == 20)
print("\nDONE — all checks passed." if not FAILS else f"\n{len(FAILS)} FAILED")
raise SystemExit(1 if FAILS else 0)
