"""diag_connect.py - calls the backend's /connect/google-drive directly, the way the website does. Prints status and body,
never the secret. Temporary: delete it when the Drive import works.   Run: python diag_connect.py"""
import json
import os
import re
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
secret = ""
for line in open(os.path.join(HERE, ".env"), encoding="utf-8"):
    m = re.match(r"\s*PYTHON_SERVICE_SECRET\s*=\s*(.*?)\s*$", line)
    if m:
        secret = m.group(1).strip("\"'")

def post(link):
    req = urllib.request.Request(
        "http://127.0.0.1:8000/connect/google-drive",
        data=json.dumps({"link": link, "sessionId": "diag"}).encode(),
        headers={"Content-Type": "application/json", "X-Service-Secret": secret}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return r.status, r.read().decode()[:300]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]
    except Exception as e:
        return "NETWORK ERROR", f"{type(e).__name__}: {e}"

print("secret found in .env:", bool(secret))
print("\n1) your FOLDER link (expect: 200 and 'That link is to a folder...')")
print("  ", post("https://drive.google.com/drive/folders/1GKJLXG95j0cyvRCpgCVYsUJawU_gl6CC?usp=sharing"))
print("\n2) a file link with a made-up id (expect: 200 and a plain-English failure; this tests the backend reaching Google)")
print("  ", post("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz012345/view?usp=sharing"))
