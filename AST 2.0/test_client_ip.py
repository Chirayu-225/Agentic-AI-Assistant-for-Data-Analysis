"""
test_client_ip.py — the caller address used for rate limiting must not be spoofable.

X-Forwarded-For is "client, proxy1, proxy2": each trusted proxy appends the address it saw,
so only entries counted from the RIGHT can be trusted. The first entry is attacker-controlled.

Run: python3 test_client_ip.py
"""
import os, sys, tempfile

os.environ["AUDIT_LOG_PATH"] = os.path.join(tempfile.mkdtemp(prefix="ip_test_"), "audit.jsonl")
os.environ["PYTHON_SERVICE_SECRET"] = "s3cret-for-tests"

import api

FAIL = []
def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(label)

class _Req:
    def __init__(self, xff="", host="9.9.9.9"):
        self.headers = {"x-forwarded-for": xff} if xff else {}
        self.client = type("C", (), {"host": host})()

def ip(req, hops):
    api.TRUSTED_PROXY_HOPS = hops
    return api._client_ip(req)

check("one trusted proxy: a spoofed first entry is ignored", ip(_Req("6.6.6.6, 1.2.3.4"), 1) == "1.2.3.4")
check("one trusted proxy: single appended entry is used", ip(_Req("1.2.3.4"), 1) == "1.2.3.4")
check("rotating the fake first entry cannot change the result",
      len({ip(_Req(f"10.0.0.{i}, 1.2.3.4"), 1) for i in range(20)}) == 1)
check("two trusted proxies: second from the right", ip(_Req("6.6.6.6, 1.2.3.4, 10.0.0.1"), 2) == "1.2.3.4")
check("more hops than entries: falls back to the left-most entry", ip(_Req("1.2.3.4"), 3) == "1.2.3.4")
check("no header: socket peer address", ip(_Req(""), 1) == "9.9.9.9")
check("hops=0 (no proxy): header ignored entirely", ip(_Req("6.6.6.6"), 0) == "9.9.9.9")

if FAIL:
    print(f"\n{len(FAIL)} FAILED"); sys.exit(1)
print("\nall passed")
