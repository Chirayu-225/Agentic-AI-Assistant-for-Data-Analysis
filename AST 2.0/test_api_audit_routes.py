"""The three audit routes end to end through FastAPI: auth, bad input, the trail, the CSV and the duplicate choice.
Skips (exit 0) when fastapi is not installed. Run: python test_api_audit_routes.py"""
import os, sys, tempfile

try:
    from fastapi.testclient import TestClient
except Exception:
    print("[SKIP] fastapi not installed")
    sys.exit(0)

TMP = tempfile.mkdtemp(prefix="api_audit_test_")
os.environ["AUDIT_LOG_PATH"] = os.path.join(TMP, "audit.jsonl")
os.environ["PYTHON_SERVICE_SECRET"] = "s3cret-for-tests"
for k in ("RATE_LIMIT_ENABLED", "RATE_LIMIT_MULTIPLIER", "CACHE_ENABLED"):
    os.environ.pop(k, None)

import api, local_store
os.environ["PYTHON_SERVICE_SECRET"] = "s3cret-for-tests"
api.SERVICE_SECRET = "s3cret-for-tests"

FAIL = []
def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" - {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(label)

H = {"x-service-secret": "s3cret-for-tests"}
rows = ["order_id,customer,amount,state"]
for i in range(60):
    rows.append(f"ORD-{i:04d},C{i % 7},{10 + i},{'CA' if i % 2 else 'NY'}")
rows.append(rows[6])  # one exact copy of an earlier row
raw = ("\n".join(rows) + "\n").encode()

url = local_store.save_bytes(raw)
c = TestClient(api.app)
body = {"fileUrl": url, "filename": "t.csv", "sessionId": "s-audit-1"}

for path in ("/clean/audit", "/clean/audit/csv", "/clean/duplicates"):
    r = c.post(path, json=body)
    check(f"{path}: no secret -> rejected", r.status_code in (401, 403), r.status_code)

r = c.post("/clean/audit", json={**body, "fileUrl": "https://evil.example/x.csv"}, headers=H)
check("remote url refused", r.status_code == 200 and r.json().get("success") is False)

r = c.post("/clean/audit", json=body, headers=H)
j = r.json()
check("audit ok", r.status_code == 200 and j.get("success") is True, str(j)[:200])
check("audit has summary + entries", "summary" in j and isinstance(j.get("entries"), list))
check("the cleaner's removal of the ID-proven copy is in the trail",
      j.get("summary", {}).get("totals", {}).get("rowsRemoved") == 1, str(j.get("summary", {}).get("totals")))
check("nothing unexplained", j.get("summary", {}).get("unexplained", {}).get("count") == 0)
check("no traceback or path in response", "Traceback" not in r.text and "C:\\" not in r.text)

r = c.post("/clean/audit/csv", json=body, headers=H)
check("csv ok", r.status_code == 200 and "text/csv" in r.headers.get("content-type", ""))

r = c.post("/clean/duplicates", json=body, headers=H)
j = r.json()
check("duplicates endpoint ok", r.status_code == 200 and j.get("success") is True, str(j)[:200])
# The copy repeats its order_id, so the cleaner already removed it (an ID proves it is a copy): nothing is left to choose.
check("no repeats left for the keep/remove choice", j.get("removed") == 0 and j.get("rows") == 60, str(j))

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nAll checks passed.")
sys.exit(1 if FAIL else 0)
