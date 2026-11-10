"""
detect_encoding.py — the "encoding artifacts" check: text that was decoded with the wrong character set.

A pure function: DataFrame in, cards out. No I/O, no LLM, no state.

WHAT IT FLAGS (text columns only; blank cells are skipped)
    c1_control   the value contains a C1 control character, U+0080 to U+009F. These never occur in real text;
                 they appear when bytes from a Windows-1252 file are decoded as latin-1 (0x84 -> U+0084).
    mojibake     UTF-8 text read as Windows-1252 / latin-1: "Ã" followed by a continuation glyph (Ã¤, Ã©, Ã¶, Ã¼),
                 "Â" followed by U+00A0 to U+00BF (Â°, Â£), or "â€" (the start of a curly quote, dash or ellipsis).
    Genuine accented text (é ä ö ü å ñ ç, "SÃO PAULO", "pâté", "5€", curly quotes) is not flagged.

ONE CARD PER COLUMN
    {"column", "reason": "encoding_artifact", "count", "of", "kinds", "example"}
      count    values in the column that contain an artifact
      of       non-blank values in the column
      kinds    ["c1_control"], ["mojibake"] or both, in that order
      example  the first hit, with control characters escaped (\\x84), cut to EXAMPLE_MAX characters

LIMITS (shown, not hidden)
    * A file whose wrong encoding still decodes to clean-looking text (for example a reader that maps byte 0x84 to
      the printable "„") is NOT caught here. Only the two signatures above are.
    * Other encodings' mojibake (Mac Roman, Cyrillic, ...) is not covered.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

REASON = "encoding_artifact"
EXAMPLE_MAX = 60

_C1 = re.compile("[\u0080-\u009f]")
# UTF-8 read as cp1252 / latin-1: lead "Ã" or "Â" followed by a continuation glyph, or "â€" (curly quotes, dashes, ellipsis)
_CONT = ("\u0080-¿ŒœŠšŸŽžƒˆ˜–—‘-„"
         "†-•…‰‹›€™")
_MOJIBAKE = re.compile(f"Ã[{_CONT}]|Â[ -¿]|â(?:€|\u0080)")

_NAMED = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _escape(text: str) -> str:
    """Make control characters visible (\\x84) and keep every printable character as it is."""
    out = []
    for ch in text:
        if ch in _NAMED:
            out.append(_NAMED[ch])
        elif unicodedata.category(ch).startswith("C"):
            out.append(f"\\x{ord(ch):02x}" if ord(ch) < 0x100 else f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return "".join(out)


def _is_text_column(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def find_encoding_artifacts(df: pd.DataFrame) -> list[dict]:
    """One card per text column that contains an encoding artifact, in column order. [] for none / no columns."""
    cards: list[dict] = []
    if df is None or df.shape[1] == 0:
        return cards
    for i in range(df.shape[1]):
        series = df.iloc[:, i]          # by position: safe even if two columns share a name
        if not _is_text_column(series):
            continue
        text = series.dropna().astype(str)
        text = text[text.str.strip() != ""]
        if text.empty:
            continue
        c1 = text.str.contains(_C1, regex=True)
        moji = text.str.contains(_MOJIBAKE, regex=True)
        hit = c1 | moji
        if not hit.any():
            continue
        kinds = [name for name, mask in (("c1_control", c1), ("mojibake", moji)) if mask.any()]
        cards.append({
            "column": df.columns[i],
            "reason": REASON,
            "count": int(hit.sum()),
            "of": int(len(text)),
            "kinds": kinds,
            "example": _escape(text[hit].iloc[0])[:EXAMPLE_MAX],
        })
    return cards
