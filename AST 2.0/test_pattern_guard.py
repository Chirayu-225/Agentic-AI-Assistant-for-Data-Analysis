"""test_pattern_guard.py — the string-level guard must not reject legitimate column
names, and must still reject real attacks.   Run: python3 test_pattern_guard.py"""
import sys
from sandbox import is_code_safe, is_code_safe_patterns
FAIL=[]
def check(l,c,d=""):
    print(f"[{'OK' if c else 'FAIL'}] {l}"+(f" — {d}" if d and not c else ""))
    if not c: FAIL.append(l)

def main():
    ok = {
     "requests column":   "x = df['total_of_special_requests'].mean()",
     "requests attr col": "df['special_requests'] = df['special_requests'].fillna(0)",
     "socket column":     "df['socket_type'] = df['socket_type'].str.strip()",
     "pos. suffix":       "y = df['pos'].str.lower(); cos = df['cos'] * 2",
     "reopen name":       "df['reopen_date'] = pd.to_datetime(df['reopen_date'])",
     "sys in text":       "df['note'] = df['note'].replace('sys.admin', 'admin')",
     "comment mention":   "# do not use os.system here\nx = df['a'].sum()",
     "identifier requests_per_day": "requests_per_day = df['a'] / 30",
     "multiline string":  'x = df["a"].replace("""os.\nsystem""", "")',
    }
    for n,c in ok.items():
        s,r=is_code_safe(c); check(f"allowed: {n}", s, r)
    bad = {
     "os.system":      "os.system('dir')",
     "requests.get":   "requests.get('http://x')",
     "open call":      "open('x.txt')",
     "eval call":      "eval('1+1')",
     "getattr":        "getattr(df, 'to_csv')",
     "subprocess":     "subprocess.run(['dir'])",
     "while True":     "while True:\n    pass",
     "dunder in str":  "x = df['__class__']",
     "fstring os":     'x = f"{os.getcwd()}"',
     "import":         "import os",
     "socket module":  "socket.socket()",
     "sys.exit":       "sys.exit(1)",
     "exec spaced":    "exec ('1')",
    }
    for n,c in bad.items():
        s,r=is_code_safe(c); check(f"blocked: {n}", not s)
    print()
    if FAIL:
        print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}"); sys.exit(1)
    print("DONE — all checks passed.")
if __name__=="__main__": main()
