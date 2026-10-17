"""
test_local_upload.py — POST /upload + local:// ids: the no-Vercel-Blob path.
Covers auth, streaming a large file, content-addressing, size cap, empty files,
path-traversal ids, expiry, and that every endpoint family can read a local file.

Run: python3 test_local_upload.py
"""
import io, os, shutil, sys, tempfile, time

TMP = tempfile.mkdtemp(prefix="upload_test_")
os.environ["LOCAL_UPLOAD_DIR"] = os.path.join(TMP, "uploads")
os.environ["AUDIT_LOG_DISABLED"] = "1"
os.environ["PYTHON_SERVICE_SECRET"] = "s3cret-for-tests"
for k in ("RATE_LIMIT_ENABLED", "RATE_LIMIT_MULTIPLIER", "CACHE_ENABLED", "MAX_UPLOAD_MB", "LOCAL_UPLOAD_TTL_HOURS"):
    os.environ.pop(k, None)

from fastapi.testclient import TestClient
import api, cache_layer, rate_limiter, local_store

FAIL = []
def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(label)

H = {"x-service-secret": "s3cret-for-tests", "x-filename": "sales.csv"}
CSV = b"region,units,price\nNorth,10,2.5\nSouth,20,3.5\nEast,30,4.5\n"

def main():
    try:
        rate_limiter.limiter.reset(); cache_layer.clear_all()
        c = TestClient(api.app)

        print("=" * 70); print("PART 1 — auth, basic round trip, content addressing"); print("=" * 70)
        r = c.post("/upload", content=CSV)
        check("upload without secret -> 401", r.status_code == 401)
        r = c.post("/upload", content=CSV, headers={**H, "x-service-secret": "wrong"})
        check("upload with wrong secret -> 401", r.status_code == 401)

        r = c.post("/upload", content=CSV, headers=H)
        j = r.json()
        check("upload succeeds", r.status_code == 200 and j.get("success") is True, r.text)
        url = j.get("url", "")
        check("returns a local:// id", url.startswith("local://") and len(url) == len("local://") + 32, url)
        check("reports the byte count", j.get("bytes") == len(CSV))
        r2 = c.post("/upload", content=CSV, headers=H)
        check("same bytes -> same id (content addressed)", r2.json().get("url") == url)
        r3 = c.post("/upload", content=CSV + b"West,5,1.0\n", headers=H)
        check("different bytes -> different id", r3.json().get("url") != url)

        print("\n" + "=" * 70); print("PART 2 — every endpoint family can read a local file"); print("=" * 70)
        r = c.post("/profile", json={"fileUrl": url, "filename": "sales.csv"}, headers=H)
        j = r.json()
        check("/profile reads the uploaded file", j.get("success") and j.get("totalRows") == 3, r.text[:200])
        r = c.post("/clean", json={"fileUrl": url, "filename": "sales.csv"}, headers=H)
        check("/clean reads the uploaded file", r.json().get("success") is True, r.text[:200])
        r = c.post("/multi-table/query", json={"datasets": [{"alias": "s", "fileUrl": url, "filename": "sales.csv"}],
                                              "query": "x", "sessionId": "t"}, headers=H)
        check("multi-table fetch path accepts local ids (no 'fetch' error)",
              "Could not fetch" not in r.text and "no longer available" not in r.text, r.text[:200])

        print("\n" + "=" * 70); print("PART 3 — large file streams through"); print("=" * 70)
        big = b"id,v\n" + b"".join(b"%d,%d\n" % (i, i * 2) for i in range(2_500_000))  # ~ 30 MB
        t0 = time.perf_counter()
        r = c.post("/upload", content=big, headers=H)
        dt = time.perf_counter() - t0
        check(f"{len(big)/1e6:.0f} MB upload accepted ({dt:.1f}s)", r.status_code == 200 and r.json().get("bytes") == len(big), r.text[:200])
        bigurl = r.json().get("url", "")
        check("stored bytes are byte-identical", local_store.load(bigurl) == big)

        print("\n" + "=" * 70); print("PART 4 — limits and failure modes"); print("=" * 70)
        before = set(os.listdir(local_store.upload_dir()))
        os.environ["MAX_UPLOAD_MB"] = "0.0005"          # ~524 bytes
        r = c.post("/upload", content=b"x" * 5000, headers=H)
        check("over the cap -> 413", r.status_code == 413, str(r.status_code))
        after = set(os.listdir(local_store.upload_dir()))
        check("a rejected upload leaves no partial file behind", after == before, str(after - before))
        del os.environ["MAX_UPLOAD_MB"]

        r = c.post("/upload", content=b"", headers=H)
        check("empty upload -> 400", r.status_code == 400, str(r.status_code))
        check("empty upload leaves no partial file", not [f for f in os.listdir(local_store.upload_dir()) if f.endswith(".part")])

        print("\n" + "=" * 70); print("PART 5 — ids cannot escape the upload folder"); print("=" * 70)
        for bad in ("local://../../etc/passwd", "local://..%2f..%2fsecret", "local://" + "g" * 32,
                    "local://" + "a" * 31, "local://" + "a" * 33, "local://", "local://a/b"):
            r = c.post("/profile", json={"fileUrl": bad, "filename": "a.csv"}, headers=H)
            check(f"malformed id rejected: {bad[:34]}", r.json().get("success") is False and "no longer available" in r.text, r.text[:120])
        r = c.post("/profile", json={"fileUrl": "local://" + "0" * 32, "filename": "a.csv"}, headers=H)
        check("unknown but well-formed id -> clear 're-upload' message", "upload it again" in r.text, r.text[:120])
        check("load() returns None for traversal ids", local_store.load("local://../../etc/passwd") is None)

        print("\n" + "=" * 70); print("PART 6 — expiry and cleanup"); print("=" * 70)
        p = local_store._path_for(url[len("local://"):])
        old = time.time() - 48 * 3600
        os.utime(p, (old, old))
        check("a file older than the TTL is treated as expired", local_store.load(url) is None)
        removed = local_store.cleanup_old()
        check("cleanup removes expired files", removed >= 1 and not os.path.exists(p), str(removed))
        r = c.post("/upload", content=CSV, headers=H)
        check("re-uploading an expired file works again", r.json().get("url") == url and local_store.load(url) == CSV)

        print("\n" + "=" * 70); print("PART 7 — Excel round trip"); print("=" * 70)
        import pandas as pd
        buf = io.BytesIO(); pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_excel(buf, index=False)
        r = c.post("/upload", content=buf.getvalue(), headers={**H, "x-filename": "t.xlsx"})
        r = c.post("/profile", json={"fileUrl": r.json()["url"], "filename": "t.xlsx"}, headers=H)
        check("xlsx uploaded and profiled", r.json().get("success") and r.json().get("totalRows") == 3, r.text[:200])
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

    print()
    if FAIL:
        print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}"); sys.exit(1)
    print("DONE — all checks passed.")

if __name__ == "__main__":
    main()
