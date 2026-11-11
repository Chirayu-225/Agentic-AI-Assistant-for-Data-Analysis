"""
test_drive_folder.py — importing several files from Google Drive: a folder link, or several links.

Google is replaced by a fake HTTP client that answers by URL, so no network is used. What must hold: a single file link still
behaves exactly as before; a folder is listed (official API with a key, public embedded view without) and its CSV / Excel /
Google Sheet files imported; one bad file never sinks the rest; limits apply; and the API key never appears anywhere the user
can see. Run: python test_drive_folder.py
"""
import os
import tempfile

os.environ["LOCAL_UPLOAD_DIR"] = tempfile.mkdtemp(prefix="df_test_")
os.environ.pop("GOOGLE_API_KEY", None)
os.environ.pop("DRIVE_FOLDER_MAX_FILES", None)
os.environ.pop("DRIVE_IMPORT_BUDGET_S", None)

import re

import data_connectors as dc
import local_store

FAIL = []
KEY = "AIzaSyFAKE-KEY-that-must-never-leak-0123456789"


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except dc.ConnectorError as e:
        return str(e)
    return None


def fid(n):
    return f"FileId{n:02d}AbCdEfGhIjKlMnOpQr"[:33].ljust(33, "x")


FOLDER = "1FolderAbCdEfGhIjKlMnOpQrStUvWx"
CSV = b"id,name,amount\n1,a,10\n2,b,20\n3,c,30\n"
XLSX = b"PK\x03\x04" + b"0" * 200


class Resp:
    def __init__(self, status=200, body=b"", headers=None, json_body=None):
        self.status_code, self.content, self.headers, self._json = status, body, headers or {}, json_body
        self.closed = False

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json

    def close(self):
        self.closed = True


class Net:
    """A fake Google. `listing` is the files.list body; `embed` the embedded-view HTML; `blobs` maps id -> bytes (or an int
    status to fail with); `sheets` maps id -> csv bytes."""

    def __init__(self, listing=None, embed=None, blobs=None, sheets=None, list_status=200, list_json=None, boom=None):
        self.listing, self.embed, self.blobs, self.sheets = listing, embed, blobs or {}, sheets or {}
        self.list_status, self.list_json, self.boom = list_status, list_json, boom
        self.calls = []

    def get(self, url, params=None, **kw):
        assert kw.get("allow_redirects") is False, "redirects must be followed by hand"
        self.calls.append((url, dict(params or {})))
        if self.boom and url.startswith(dc._API_LIST_URL):
            import requests
            raise requests.ConnectionError(f"HTTPSConnectionPool: Max retries exceeded with url: {url}?key={KEY}")
        if url.startswith(dc._API_LIST_URL):
            return Resp(self.list_status, json_body=self.list_json if self.list_json is not None else self.listing)
        if url.startswith(dc._EMBED_URL):
            return Resp(200, (self.embed or "").encode())
        m = re.match(r"https://docs\.google\.com/spreadsheets/d/([^/]+)/export", url)
        if m:
            b = self.sheets.get(m.group(1))
            return Resp(200, b) if b is not None else Resp(200, b"<!doctype html><html>sign in</html>")
        m = re.search(r"download\?id=([^&]+)", url)
        if m:
            b = self.blobs.get(m.group(1))
            if isinstance(b, int):
                return Resp(b)
            return Resp(200, b if b is not None else b"<html>nope</html>")
        return Resp(404)


def api_listing(files):
    return {"files": [{"id": i, "name": n, "mimeType": m, **({"size": str(s)} if s is not None else {})} for i, n, m, s in files]}


def main():
    # ── link helpers ───────────────────────────────────────────────────────────────────────────────────────────────
    check("folder id: /drive/folders/ID", dc.folder_id_from_link(f"https://drive.google.com/drive/folders/{FOLDER}?usp=sharing") == FOLDER)
    check("folder id: /drive/u/0/folders/ID", dc.folder_id_from_link(f"https://drive.google.com/drive/u/0/folders/{FOLDER}") == FOLDER)
    check("folder id: a file link is not a folder", dc.folder_id_from_link(f"https://drive.google.com/file/d/{fid(1)}/view") is None)
    check("folder id: a look-alike host is rejected", dc.folder_id_from_link(f"https://drive.google.com.evil.com/drive/folders/{FOLDER}") is None)
    check("folder id: a short id is rejected", dc.folder_id_from_link("https://drive.google.com/drive/folders/abc") is None)
    check("split_links: newlines, spaces, commas; blanks and repeats dropped; order kept",
          dc.split_links("a\nb  c,\n\nb ; d") == ["a", "b", "c", "d"])

    # ── one file link is untouched ─────────────────────────────────────────────────────────────────────────────────
    net = Net(blobs={fid(1): CSV})
    out = dc.import_drive_many(f"https://drive.google.com/file/d/{fid(1)}/view?usp=sharing", http=net)
    check("a single file link behaves exactly as before (no 'files' list)", out["success"] and "files" not in out and out["url"].startswith("local://"), str(out))
    check("one bad link is still an error, not a skip", raises(dc.import_drive_many, "https://evil.com/x", http=net) is not None)
    check("empty input is a plain error", raises(dc.import_drive_many, "   ", http=net) is not None)

    # ── a folder, official API (key set) ───────────────────────────────────────────────────────────────────────────
    os.environ["GOOGLE_API_KEY"] = KEY
    listing = api_listing([
        (fid(1), "orders.csv", "text/csv", 40), (fid(2), "customers.csv", "text/csv", 40), (fid(3), "products.xlsx",
         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 300), (fid(4), "Budget", dc._SHEET_MIME, None),
        (fid(5), "report.pdf", "application/pdf", 900), (fid(6), "Archive", dc._FOLDER_MIME, None),
        (fid(7), "Notes", "application/vnd.google-apps.document", None)])
    net = Net(listing=listing, blobs={fid(1): CSV, fid(2): CSV + b"4,d,40\n", fid(3): XLSX}, sheets={fid(4): b"a,b\n1,2\n3,4\n"})
    out = dc.import_drive_many(f"https://drive.google.com/drive/folders/{FOLDER}?usp=sharing", http=net)
    names = [f["filename"] for f in out["files"]]
    check("API folder: the four spreadsheets are imported, in listing order", names == ["orders.csv", "customers.csv", "products.xlsx", "Budget.csv"], str(names))
    check("API folder: the pdf, the subfolder and the Google Doc are skipped with a reason",
          {s["name"] for s in out["skipped"]} == {"report.pdf", "Archive", "Notes"} and all(s["reason"] for s in out["skipped"]), str(out["skipped"]))
    check("API folder: the first file is also at the top level (old clients keep working)", out["url"] == out["files"][0]["url"] and out["filename"] == "orders.csv")
    check("API folder: the Google Sheet went through the sheet export, files through the download endpoint",
          any("spreadsheets/d/" + fid(4) in c[0] for c in net.calls) and any("download?id=" + fid(1) in c[0] for c in net.calls))
    check("API folder: the folder id and the key went only to the Drive API host, as parameters",
          net.calls[0][0] == dc._API_LIST_URL and net.calls[0][1]["key"] == KEY and FOLDER in net.calls[0][1]["q"])
    check("API folder: the stored files read back unchanged", all(local_store.load(f["url"]) for f in out["files"]) and local_store.load(out["files"][0]["url"]) == CSV)
    check("API folder: the note says what was imported and skipped", out["note"].startswith("Imported 4 files.") and "Skipped 3" in out["note"], out["note"])
    check("the key is nowhere in the result", KEY not in repr(out))

    # the key must not leak through errors either
    err = raises(dc.import_drive_many, f"https://drive.google.com/drive/folders/{FOLDER}", http=Net(boom=True))
    check("a network error while listing gives a plain message without the key or a URL", err is not None and KEY not in err and "http" not in err.lower(), str(err))
    for status, body, expect in [(400, {"error": {"errors": [{"reason": "keyInvalid"}]}}, "GOOGLE_API_KEY"),
                                 (403, {"error": {"errors": [{"reason": "accessNotConfigured"}]}}, "not enabled"),
                                 (404, {}, "could not find that folder"), (403, {}, "private")]:
        err = raises(dc.import_drive_many, f"https://drive.google.com/drive/folders/{FOLDER}", http=Net(list_status=status, list_json=body))
        check(f"API error {status}: a specific, key-free message", err is not None and expect.lower() in err.lower() and KEY not in err, str(err))

    # ── partial failure, caps, dedupe, time budget ─────────────────────────────────────────────────────────────────
    net = Net(listing=api_listing([(fid(1), "a.csv", "text/csv", 40), (fid(2), "b.csv", "text/csv", 40), (fid(3), "c.csv", "text/csv", 40)]),
              blobs={fid(1): CSV, fid(2): 403, fid(3): CSV + b"9,z,99\n"})
    out = dc.import_drive_many(f"https://drive.google.com/drive/folders/{FOLDER}", http=net)
    check("one private file is skipped with the sharing hint; the other two still import",
          [f["filename"] for f in out["files"]] == ["a.csv", "c.csv"] and out["skipped"][0]["name"] == "b" or out["skipped"][0]["name"] == "b.csv", str(out["skipped"]))
    check("... and the skip reason is the plain-English sharing message", "private" in out["skipped"][0]["reason"].lower(), str(out["skipped"]))

    net = Net(listing=api_listing([(fid(1), "a.csv", "text/csv", 40)]), blobs={fid(1): 403})
    err = raises(dc.import_drive_many, f"https://drive.google.com/drive/folders/{FOLDER}", http=net)
    check("every file failing is an error with the first reason", err is not None and "private" in err.lower(), str(err))

    many = api_listing([(fid(i), f"t{i}.csv", "text/csv", 40) for i in range(1, 16)])
    net = Net(listing=many, blobs={fid(i): CSV + f"{i},x,{i}\n".encode() for i in range(1, 16)})
    out = dc.import_drive_many(f"https://drive.google.com/drive/folders/{FOLDER}", http=net)
    check("a folder of 15 imports 12 and reports the other 3 as over the limit",
          out["count"] == 12 and sum("12-file limit" in s["reason"] for s in out["skipped"]) == 3, f"{out['count']} {out['skipped']}")
    os.environ["DRIVE_FOLDER_MAX_FILES"] = "2"
    out = dc.import_drive_many(f"https://drive.google.com/drive/folders/{FOLDER}", http=net)
    check("the limit is configurable", out["count"] == 2)
    del os.environ["DRIVE_FOLDER_MAX_FILES"]

    net = Net(blobs={fid(1): CSV})
    out = dc.import_drive_many(f"https://drive.google.com/file/d/{fid(1)}/view\nhttps://drive.google.com/open?id={fid(1)}\n{fid(1)}", http=net)
    check("the same file given twice (two link styles) is imported once",
          out.get("count", 1) == 1 or len(out.get("files", [1])) == 1, str(out)[:200])

    os.environ["DRIVE_IMPORT_BUDGET_S"] = "0"
    net = Net(listing=api_listing([(fid(1), "a.csv", "text/csv", 40), (fid(2), "b.csv", "text/csv", 40)]), blobs={fid(1): CSV, fid(2): CSV})
    err = raises(dc.import_drive_many, f"https://drive.google.com/drive/folders/{FOLDER}", http=net)
    check("a spent time budget skips the remaining files with a clear reason", err is not None and "out of time" in err.lower(), str(err))
    del os.environ["DRIVE_IMPORT_BUDGET_S"]

    big = api_listing([(fid(1), "huge.csv", "text/csv", 10 ** 12), (fid(2), "ok.csv", "text/csv", 40)])
    out = dc.import_drive_many(f"https://drive.google.com/drive/folders/{FOLDER}", http=Net(listing=big, blobs={fid(2): CSV}))
    check("a file whose listed size is over the limit is skipped before any download",
          [f["filename"] for f in out["files"]] == ["ok.csv"] and "MB limit" in out["skipped"][0]["reason"])

    # ── several links, mixed ───────────────────────────────────────────────────────────────────────────────────────
    net = Net(listing=api_listing([(fid(1), "a.csv", "text/csv", 40)]), blobs={fid(1): CSV, fid(2): CSV + b"7,q,70\n"})
    text = f"https://drive.google.com/drive/folders/{FOLDER}\nhttps://drive.google.com/file/d/{fid(2)}/view\nhttps://evil.com/x"
    out = dc.import_drive_many(text, http=net)
    check("a folder link and a file link together import both; the bad link is skipped, not fatal",
          out["count"] == 2 and any(s["name"] == "link 3" for s in out["skipped"]), str(out["skipped"]))
    out = dc.import_drive_many(f"https://drive.google.com/file/d/{fid(1)}/view\nhttps://drive.google.com/file/d/{fid(2)}/view", http=net)
    check("several file links import all of them", out["count"] == 2 and "files" in out)

    # ── no key: the public embedded view ───────────────────────────────────────────────────────────────────────────
    del os.environ["GOOGLE_API_KEY"]
    html = ("<html><body>"
            f'<div class="flip-entry" id="entry-{fid(1)}"><a href="x"><div class="flip-entry-title">sales &amp; costs.csv</div></a></div>'
            f'<div class="flip-entry" id="entry-{fid(2)}"><a href="x"><div class="flip-entry-title">Team Sheet</div></a></div>'
            f'<div class="flip-entry" id="entry-{fid(3)}"><a href="x"><div class="flip-entry-title">Meeting notes</div></a></div>'
            f'<div class="flip-entry" id="entry-{fid(4)}"><a href="x"><div class="flip-entry-title">logo.png</div></a></div>'
            "</body></html>")
    net = Net(embed=html, blobs={fid(1): CSV}, sheets={fid(2): b"x,y\n1,2\n3,4\n"})
    out = dc.import_drive_many(f"https://drive.google.com/drive/folders/{FOLDER}", http=net)
    check("no key: the embedded view lists the entries; a csv and a Google Sheet (no extension) import",
          [f["filename"] for f in out["files"]] == ["sales _ costs.csv", "Team Sheet.csv"], str([f["filename"] for f in out["files"]]))
    check("no key: an extensionless doc that is not a sheet, and a png, are skipped with a reason",
          {s["name"] for s in out["skipped"]} == {"Meeting notes", "logo.png"} and all(s["reason"] for s in out["skipped"]), str(out["skipped"]))
    check("no key: only the fixed embedded-view host was asked to list the folder", net.calls[0][0] == dc._EMBED_URL and net.calls[0][1] == {"id": FOLDER})
    err = raises(dc.import_drive_many, f"https://drive.google.com/drive/folders/{FOLDER}", http=Net(embed="<html>sign in</html>"))
    check("no key, a page with no entries: a plain 'check the sharing' message", err is not None and "shared" in err.lower(), str(err))

    # ── wiring ─────────────────────────────────────────────────────────────────────────────────────────────────────
    here = os.path.dirname(os.path.abspath(__file__))
    api_src = open(os.path.join(here, "api.py"), encoding="utf-8").read()
    check("api.py calls import_drive_many and audits the file count", "import_drive_many(req.link)" in api_src and "files=" in api_src)

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        raise SystemExit(1)
    print("DONE — all checks passed.")


if __name__ == "__main__":
    main()
