"""
test_api_hardening.py — the real FastAPI app, driven through TestClient with
file downloads stubbed: shared-secret auth (+audit), X-Request-ID, per-session
429 + Retry-After, per-IP backstop, blob/result caching, piiReport in /profile,
/admin/stats, and that the audit log never contains file contents.

Run: python3 test_api_hardening.py
"""
import json, os, shutil, sys, tempfile

TMP = tempfile.mkdtemp(prefix="api_test_")
LOG = os.path.join(TMP, "audit.jsonl")
os.environ["AUDIT_LOG_PATH"] = LOG
os.environ.pop("AUDIT_LOG_DISABLED", None)
os.environ["PYTHON_SERVICE_SECRET"] = "s3cret-for-tests"
for k in ("RATE_LIMIT_ENABLED", "RATE_LIMIT_MULTIPLIER", "CACHE_ENABLED"):
    os.environ.pop(k, None)

from fastapi.testclient import TestClient
import api, audit_log, cache_layer, rate_limiter

FAIL = []
def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(label)

CSV = (b"customer_name,email,city,amount\n"
       b"Asha Rao,asha.secret@example.com,Pune,120\n"
       b"Vikram Shah,vikram@example.org,Delhi,300\n"
       b"Neha Iyer,neha@example.in,Pune,99\n")
fetches = {"n": 0}
class FakeResp:
    def __init__(self, c): self.content = c
    def raise_for_status(self): pass
def fake_get(url, timeout=None):
    fetches["n"] += 1
    if "dead" in url: raise ConnectionError("unreachable")
    return FakeResp(CSV)
api.requests.get = fake_get

H = {"x-service-secret": "s3cret-for-tests"}
def log_records():
    with open(LOG, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    # Guarded: on Windows the sandbox uses "spawn", which re-imports this file in
    # each worker; without the guard the whole test would re-run inside the worker.
    try:
        audit_log.reset_for_tests(); rate_limiter.limiter.reset(); cache_layer.clear_all()
        c = TestClient(api.app)

        print("=" * 70); print("PART 1 — auth and request ids"); print("=" * 70)
        r = c.get("/health")
        check("/health needs no secret", r.status_code == 200)
        r = c.post("/profile", json={"fileUrl": "https://blob/x1", "filename": "a.csv"})
        check("missing secret -> 401", r.status_code == 401)
        r = c.post("/profile", json={"fileUrl": "https://blob/x1", "filename": "a.csv"}, headers={"x-service-secret": "wrong"})
        check("wrong secret -> 401", r.status_code == 401)
        check("401 responses still carry X-Request-ID", bool(r.headers.get("x-request-id")))
        ev = [x["event"] for x in log_records()]
        check("auth failures are audited", ev.count("auth_failure") == 2, str(ev))
        check("the audit log never contains the secret or the wrong guess",
              "s3cret-for-tests" not in open(LOG).read() and '"wrong"' not in open(LOG).read())

        print("\n" + "=" * 70); print("PART 2 — /profile: PII report, caching"); print("=" * 70)
        body = {"fileUrl": "https://blob/file-1", "filename": "customers.csv", "sessionId": "sess-A"}
        r1 = c.post("/profile", json=body, headers=H)
        j = r1.json()
        check("profile succeeds", r1.status_code == 200 and j.get("success"), str(j)[:200])
        pr = j.get("piiReport") or {}
        cols = {f["column"] for f in pr.get("flaggedColumns", [])}
        check("piiReport flags name and email", {"customer_name", "email"} <= cols, str(cols))
        check("piiReport does not flag city/amount", not ({"city", "amount"} & cols))
        check("request id is returned", bool(r1.headers.get("x-request-id")))
        n_after_first = fetches["n"]
        r2 = c.post("/profile", json=body, headers=H)
        check("second identical request returns the same answer", r2.json() == j)
        check("second request did not re-download the file (blob cache)", fetches["n"] == n_after_first)
        check("result cache recorded a hit", cache_layer.stats()["result"]["hits"] >= 1)
        r3 = c.post("/profile", json={**body, "filename": "renamed.csv"}, headers=H)
        check("a different filename is not served a stale echo", r3.json().get("filename", "renamed.csv") == "renamed.csv")
        d = c.post("/profile", json={"fileUrl": "https://blob/dead", "filename": "d.csv"}, headers=H).json()
        check("an unreachable file gives a clean error, not a 500", d.get("success") is False and "Could not fetch" in d.get("error", ""))

        print("\n" + "=" * 70); print("PART 3 — rate limiting (real 429s)"); print("=" * 70)
        rate_limiter.limiter.reset()
        codes = [c.post("/clean", json={"fileUrl": "https://blob/rl", "filename": "a.csv"}, headers=H).status_code for _ in range(32)]
        check("light policy: first 30/min pass, then 429", codes[:30].count(200) == 30 and 429 in codes[30:], str(codes))
        r = c.post("/clean", json={"fileUrl": "https://blob/rl", "filename": "a.csv"}, headers=H)
        check("429 carries Retry-After", r.status_code == 429 and int(r.headers.get("retry-after", 0)) >= 1)
        check("429 body is a readable error", "Too many requests" in json.dumps(r.json()))
        r = c.post("/clean", json={"fileUrl": "https://blob/other-file", "filename": "a.csv"}, headers=H)
        check("a different caller/file key is unaffected", r.status_code == 200)
        check("rate_limited events are audited", any(x["event"] == "rate_limited" for x in log_records()))

        # The heavy endpoint is stubbed and the limiter's clock frozen: the real
        # pipeline can take many seconds per request (especially where the
        # sandbox starts workers with "spawn", as on Windows), which would let
        # the 60-second window slide mid-test and make this check flaky.
        rate_limiter.limiter.reset()
        real_auto, real_clock = api.run_auto_analyze, rate_limiter.limiter._clock
        api.run_auto_analyze = lambda content, filename, session_id: {"success": True}
        rate_limiter.limiter._clock = lambda: 1000.0
        try:
            codes = [c.post("/auto-analyze", json={"sessionId": "hammer", "fileUrl": "https://blob/h", "filename": "a.csv"},
                            headers=H).status_code for _ in range(6)]
        finally:
            api.run_auto_analyze, rate_limiter.limiter._clock = real_auto, real_clock
        check("heavy endpoint: 429 after 4 per minute per session", codes[:4] == [200] * 4 and codes[4:] == [429, 429], str(codes))

        rate_limiter.limiter.reset()
        codes = [c.get("/admin/stats", headers=H).status_code for _ in range(305)]
        check("per-IP backstop trips at 300/min", codes[299] == 200 and codes[-1] == 429, f"{codes[298:302]}")
        rate_limiter.limiter.reset()
        check("/health is exempt from the IP limit", all(c.get("/health").status_code == 200 for _ in range(310)))

        print("\n" + "=" * 70); print("PART 4 — /admin/stats and audit privacy"); print("=" * 70)
        rate_limiter.limiter.reset()
        check("/admin/stats requires the secret", c.get("/admin/stats").status_code == 401)
        s = c.get("/admin/stats", headers=H).json()
        check("stats has cache / rateLimiter / audit sections", {"cache", "rateLimiter", "audit"} <= set(s))
        check("audit chain verifies as intact after all of the above", s["audit"]["chain"]["ok"], str(s["audit"]["chain"]))
        check("event counts include request and rate_limited",
              s["audit"]["eventCounts"].get("request", 0) > 0 and s["audit"]["eventCounts"].get("rate_limited", 0) > 0)
        raw = open(LOG, encoding="utf-8").read()
        for secret in ("asha.secret@example.com", "Asha Rao", "vikram@example.org", "https://blob/file-1"):
            check(f"audit log never contains {secret!r}", secret not in raw)
        check("requests carry method/path/status/duration",
              any(x["event"] == "request" and x.get("path") == "/profile" and x.get("status") == 200 and "duration_ms" in x for x in log_records()))
    finally:
        audit_log.reset_for_tests(); shutil.rmtree(TMP, ignore_errors=True)

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}"); sys.exit(1)
    print("DONE — all checks passed.")



if __name__ == "__main__":
    main()
