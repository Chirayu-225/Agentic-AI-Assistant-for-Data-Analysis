"""
error_scrub.py — keep tracebacks, file paths and file names out of any text a user can see.

Pure standard library; it imports nothing from this project, so any layer can use it.

    scrub_error_text(text) -> str   one error string
    scrub_result(obj)      -> obj   a service result (dict / list), every error-ish string scrubbed

WHAT scrub_error_text DOES (a non-str is returned unchanged)
  * drops a "Traceback (most recent call last):" block: the header, every  File "...", line N, in ...  frame, and the
    indented source / ^^^^ lines under it. The exception line that ends the block ("KeyError: 'sales'") is kept, so
    the feedback that helps (a missing column, a bad value) stays. Text before the header on the same line stays.
  * removes absolute Windows paths (C:\\Users\\x\\secret.py, C:/x/y.csv; folders with spaces are fine), POSIX paths
    ending in .py, POSIX paths under the usual system folders (/tmp, /home, /var, ...), any  File "<...>"  fragment, and
    bare *.py file names. A path becomes "[path]", a bare .py name "[file]", so the sentence stays readable.
  * caps the result at 500 characters (ending in "...") and is idempotent: scrubbing twice gives the same text.
  Short exception lines like "KeyError: 'sales'" or "ValueError: Column not found" come out unchanged.

WHAT scrub_result DOES
  Returns a NEW structure (the input is never changed). A string is scrubbed when its key contains "error" in any
  case (error, charts_error, chartsError, lastError, ...), including strings inside a list or dict under such a key.
  Values under the keys in DATA_KEYS are the user's table data (resultData, preview, data): they are passed through by
  reference and never rewritten, because a column called "error_rate" must not have its cells cut or censored.
"""

from __future__ import annotations

import re

MAX_CHARS = 500
ELLIPSIS = "..."
PATH_PLACEHOLDER = "[path]"
FILE_PLACEHOLDER = "[file]"
DATA_KEYS = frozenset({"resultData", "preview", "data"})

_HEADER = re.compile(r"Traceback \(most recent call last\):")
_FRAME = re.compile(r'^\s*File "')
_CHAIN = re.compile(r"^\s*(?:During handling of the above exception, another exception occurred:"
                    r"|The above exception was the direct cause of the following exception:)\s*$")

# File "x.py", line 3, in run     (also File "<string>", line 1)
_FILE_FRAGMENT = re.compile(r' ?File "[^"]*"(?:, line \d+)?(?:, in [^\s,]+)?')

# C:\a\b\c.py  /  C:/a/b/c.csv  /  D:\my folder\x.txt.  Folders may contain spaces (up to 60 characters each); the last
# part may not, so the prose after a path is not swallowed. Not preceded by a letter or digit, so "http://" is not a drive.
_WINDOWS_PATH = re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/](?:[^\\/:*?"<>|\r\n]{1,60}[\\/])*[^\\/:*?"<>|\s]*')

# /tmp/x/y.csv, /home/u/app/x.py   (system folders, any file)   or   /any/where/x.py   (any folder, .py)
_POSIX_PATH = re.compile(
    r"(?<![\w:/.~-])(?:"
    r"/(?:tmp|home|usr|var|opt|srv|app|mnt|root|etc|Users|private|workspace)(?:/[\w.\-+@~]+)+"
    r"|(?:/[\w.\-+@~]+)+\.py\b"
    r")"
)

_BARE_PY = re.compile(r"(?<![\w/\\])[\w.\-]+\.py\b")

_TRAILING = ".,;:)]}'\""


def _sub_path(match: "re.Match[str]") -> str:
    """Replace a path but give back the sentence punctuation that was glued to its end ("...x.py)." )."""
    text = match.group(0)
    kept = text.rstrip(_TRAILING)
    return PATH_PLACEHOLDER + text[len(kept):]


def _drop_traceback_blocks(text: str) -> str:
    out: list[str] = []
    skipping = False
    for line in text.split("\n"):
        if skipping:
            if _FRAME.match(line) or line[:1] in (" ", "\t"):
                continue                      # a frame, its source line or a ^^^^ marker
            skipping = False                  # the first unindented line is the exception line: keep it
        header = _HEADER.search(line)
        if header:
            before = line[: header.start()].rstrip()
            if before:
                out.append(before)
            skipping = True
            continue
        if _CHAIN.match(line):
            continue
        out.append(line)
    return "\n".join(out)


def scrub_error_text(text):
    """See the module docstring. Anything that is not a str is returned as it is."""
    if not isinstance(text, str):
        return text
    s = text.replace("\r\n", "\n")
    s = _drop_traceback_blocks(s)
    s = _FILE_FRAGMENT.sub("", s)
    s = _WINDOWS_PATH.sub(_sub_path, s)
    s = _POSIX_PATH.sub(_sub_path, s)
    s = _BARE_PY.sub(FILE_PLACEHOLDER, s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    if len(s) > MAX_CHARS:
        s = s[: MAX_CHARS - len(ELLIPSIS)].rstrip() + ELLIPSIS
    return s


def _walk(value, in_error: bool):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(key, str) and key in DATA_KEYS:
                out[key] = item               # the user's data: untouched
            else:
                flagged = in_error or (isinstance(key, str) and "error" in key.lower())
                out[key] = _walk(item, flagged)
        return out
    if isinstance(value, list):
        return [_walk(item, in_error) for item in value]
    if isinstance(value, tuple):
        return tuple(_walk(item, in_error) for item in value)
    if in_error and isinstance(value, str):
        return scrub_error_text(value)
    return value


def scrub_result(obj):
    """A copy of a service result with every error-ish string scrubbed. The input is never modified."""
    return _walk(obj, False)
