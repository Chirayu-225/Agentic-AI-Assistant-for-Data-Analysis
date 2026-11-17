"""
Strict codec for results crossing the sandbox boundary (worker -> API process).

WHY THIS EXISTS (hazard H3: generated code reaching the host)
    The sandbox worker is the process that runs untrusted, model-written code.
    Its result used to travel through a multiprocessing.Queue, which PICKLES.
    The API process then unpickled whatever bytes arrived - and unpickling can
    run arbitrary code. So an escaped worker could hand the API process a
    crafted object and execute code there, with the API's secrets and network.

    This module is the replacement: results are encoded into a small, closed
    set of types (JSON, plus Parquet for tables) and decoded by code that never
    imports or calls pickle. Anything outside the set is refused at the
    sender, and anything malformed is refused at the receiver.

WHAT CROSSES THE BOUNDARY
    None, bool, int, float, str, bytes, list, tuple, dict, set, Timestamp, Timedelta,
    date, Decimal, ndarray (NumPy's own format, pickle disabled), Index, DataFrame, Series, Plotly figures (as JSON), and a marker for
    matplotlib figures (the app already reports those as unsupported).

KNOWN, DELIBERATE TRADE-OFFS
    * Parquet columns that Arrow cannot represent (mixed-type object columns,
      Interval dtype) are converted to TEXT rather than failing. The header
      records which columns were converted.
    * The parent still parses Parquet with pyarrow. That is a far smaller
      attack surface than pickle (no code execution by design), but it is not
      zero; the container step later closes the rest.
"""
from __future__ import annotations

import datetime as _dt
import decimal as _dec
import io
import json
import struct
from typing import Any

import numpy as np
import pandas as pd

MAGIC = b"ASTR1"
MAX_DEPTH = 12
MAX_NODES = 500_000
MAX_BLOBS = 10_000
_TAGS = {"nat", "ts", "td", "date", "dec", "bytes", "df", "series", "plotly", "mpl",
         "list", "tuple", "dict", "set", "ndarray", "index"}


class CodecError(Exception):
    """The payload cannot be encoded safely, or the bytes are not our format."""


class MatplotlibFigurePlaceholder:
    """Stands in for a matplotlib Figure in the API process. The app only ever
    reports these as unsupported, so no drawing state needs to cross."""
    def savefig(self, *a, **k):
        raise CodecError("matplotlib figures are not transferred out of the sandbox")

    def gca(self, *a, **k):
        raise CodecError("matplotlib figures are not transferred out of the sandbox")


# ── encoding ────────────────────────────────────────────────────────────────
class _Encoder:
    def __init__(self) -> None:
        self.blobs: list[bytes] = []
        self.nodes = 0
        self.coerced: list[str] = []

    def _blob(self, data: bytes) -> int:
        if len(self.blobs) >= MAX_BLOBS:
            raise CodecError("too many binary parts in one result")
        self.blobs.append(data)
        return len(self.blobs) - 1

    def enc(self, v: Any, depth: int = 0) -> Any:
        self.nodes += 1
        if depth > MAX_DEPTH or self.nodes > MAX_NODES:
            raise CodecError("result is nested too deeply or has too many parts")
        if v is None or isinstance(v, (bool, str)):
            return v
        if isinstance(v, int) and not isinstance(v, np.generic):
            return v
        if isinstance(v, float) and not isinstance(v, np.generic):
            return v
        if v is pd.NaT:
            return {"$": "nat"}
        if isinstance(v, np.generic):
            if isinstance(v, np.datetime64):
                return self.enc(pd.Timestamp(v), depth + 1)
            if isinstance(v, np.timedelta64):
                return self.enc(pd.Timedelta(v), depth + 1)
            return self.enc(v.item(), depth + 1)
        if isinstance(v, pd.Timestamp):
            return {"$": "ts", "v": v.isoformat()}
        if isinstance(v, _dt.datetime):
            return {"$": "ts", "v": pd.Timestamp(v).isoformat()}
        if isinstance(v, _dt.date):
            return {"$": "date", "v": v.isoformat()}
        if isinstance(v, pd.Timedelta):
            return {"$": "td", "v": int(v.value)}
        if isinstance(v, _dec.Decimal):
            return {"$": "dec", "v": str(v)}
        if isinstance(v, (bytes, bytearray)):
            return {"$": "bytes", "b": self._blob(bytes(v))}
        if isinstance(v, np.ndarray):
            if v.dtype.hasobject:  # cannot be stored without pickle; send as a plain list
                return {"$": "list", "v": [self.enc(x, depth + 1) for x in v.tolist()]}
            buf = io.BytesIO()
            np.save(buf, v, allow_pickle=False)
            return {"$": "ndarray", "b": self._blob(buf.getvalue())}
        if isinstance(v, pd.Index):
            return {"$": "index", "multi": isinstance(v, pd.MultiIndex),
                    "v": self.enc(v.tolist(), depth + 1),
                    "names": [self.enc(n, depth + 1) for n in v.names]}
        if isinstance(v, (set, frozenset)):
            return {"$": "set", "v": [self.enc(x, depth + 1) for x in v]}
        if isinstance(v, pd.DataFrame):
            return self._frame(v, depth)
        if isinstance(v, pd.Series):
            return self._series(v, depth)
        t = str(type(v))
        if "plotly" in t and "Figure" in t:
            return {"$": "plotly", "j": v.to_json()}
        if "matplotlib" in t and "Figure" in t:
            return {"$": "mpl"}
        if isinstance(v, (list, tuple)):
            return {"$": "list" if isinstance(v, list) else "tuple",
                    "v": [self.enc(x, depth + 1) for x in v]}
        if isinstance(v, dict):
            return {"$": "dict", "v": [[self.enc(k, depth + 1), self.enc(x, depth + 1)] for k, x in v.items()]}
        raise CodecError(f"result type {type(v).__name__!r} cannot cross the sandbox boundary")

    # tables ---------------------------------------------------------------
    def _frame(self, df: pd.DataFrame, depth: int) -> dict:
        cols = df.columns
        meta: dict[str, Any] = {
            "$": "df",
            "labels": [self.enc(x, depth + 1) for x in cols.tolist()],
            "multi": isinstance(cols, pd.MultiIndex),
            "cnames": [self.enc(n, depth + 1) for n in cols.names],
        }
        idx = df.index
        body = df.copy(deep=False)
        body.columns = [f"c{k}" for k in range(body.shape[1])]
        if isinstance(idx, pd.RangeIndex):
            meta["index"] = {"kind": "range", "start": idx.start, "stop": idx.stop, "step": idx.step,
                             "name": self.enc(idx.name, depth + 1)}
        else:
            idx_frame = idx.to_frame(index=False)
            idx_frame.columns = [f"i{k}" for k in range(idx_frame.shape[1])]
            body = pd.concat([idx_frame, body.reset_index(drop=True)], axis=1)
            meta["index"] = {"kind": "levels", "n": idx.nlevels,
                             "names": [self.enc(n, depth + 1) for n in idx.names]}
        data, coerced = _to_parquet(body)
        if coerced:
            self.coerced.extend(coerced)
        meta["coerced"] = coerced
        meta["b"] = self._blob(data)
        return meta

    def _series(self, s: pd.Series, depth: int) -> dict:
        node = self._frame(s.to_frame(name="v"), depth)
        node["$"] = "series"
        node["sname"] = self.enc(s.name, depth + 1)
        return node


def _is_missing(x: Any) -> bool:
    return x is None or x is pd.NaT or (isinstance(x, float) and x != x)


def _write_parquet(body: pd.DataFrame) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(body, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _to_parquet(body: pd.DataFrame) -> tuple[bytes, list[str]]:
    """Encode a frame as Parquet. Columns Arrow/Parquet cannot represent
    (mixed-type object columns, Interval dtype, ...) are converted to text and
    reported, rather than failing the whole result or falling back to pickle."""
    import pyarrow as pa
    errors = (pa.ArrowException, TypeError, ValueError, NotImplementedError)
    try:
        return _write_parquet(body), []
    except errors:
        pass
    body = body.copy()
    coerced: list[str] = []
    for c in list(body.columns):
        try:
            _write_parquet(body[[c]])
        except errors:
            body[c] = body[c].astype(object).map(lambda x: None if _is_missing(x) else str(x))
            coerced.append(str(c))
    try:
        return _write_parquet(body), coerced
    except errors as e:
        raise CodecError(f"table cannot be encoded: {type(e).__name__}") from e


def encode_payload(payload: dict) -> bytes:
    """Serialise a worker result dict to bytes. Raises CodecError for anything
    outside the supported set - it never falls back to pickle."""
    if not isinstance(payload, dict):
        raise CodecError("payload must be a dict")
    e = _Encoder()
    try:
        header = json.dumps({"payload": e.enc(payload), "coerced": e.coerced}).encode("utf-8")
    except CodecError:
        raise
    except (TypeError, ValueError, RecursionError) as ex:
        raise CodecError(f"payload cannot be encoded: {ex}") from ex
    parts = [MAGIC, struct.pack("<I", len(header)), header, struct.pack("<I", len(e.blobs))]
    parts += [struct.pack("<Q", len(b)) for b in e.blobs]
    parts += e.blobs
    return b"".join(parts)


# ── decoding ────────────────────────────────────────────────────────────────
class _Decoder:
    def __init__(self, blobs: list[bytes]) -> None:
        self.blobs = blobs
        self.nodes = 0

    def _blob(self, i: Any) -> bytes:
        if not isinstance(i, int) or isinstance(i, bool) or not (0 <= i < len(self.blobs)):
            raise CodecError("bad binary part reference")
        return self.blobs[i]

    def dec(self, n: Any, depth: int = 0) -> Any:
        self.nodes += 1
        if depth > MAX_DEPTH or self.nodes > MAX_NODES:
            raise CodecError("result is nested too deeply or has too many parts")
        if n is None or isinstance(n, (bool, int, float, str)):
            return n
        if not isinstance(n, dict) or n.get("$") not in _TAGS:
            raise CodecError("unknown value in result")
        tag = n["$"]
        if tag == "nat":
            return pd.NaT
        if tag == "ts":
            return pd.Timestamp(str(n["v"]))
        if tag == "td":
            return pd.Timedelta(int(n["v"]), unit="ns")
        if tag == "date":
            return _dt.date.fromisoformat(str(n["v"]))
        if tag == "dec":
            return _dec.Decimal(str(n["v"]))
        if tag == "bytes":
            return self._blob(n["b"])
        if tag == "plotly":
            try:
                import plotly.io as pio
            except ImportError as e:
                raise CodecError("plotly is not installed in the API process") from e
            return pio.from_json(str(n["j"]))
        if tag == "mpl":
            return MatplotlibFigurePlaceholder()
        if tag in ("list", "tuple"):
            items = [self.dec(x, depth + 1) for x in n["v"]]
            return items if tag == "list" else tuple(items)
        if tag == "dict":
            return {self._key(self.dec(k, depth + 1)): self.dec(x, depth + 1) for k, x in n["v"]}
        if tag == "ndarray":
            arr = np.load(io.BytesIO(self._blob(n["b"])), allow_pickle=False)
            if arr.dtype.hasobject:
                raise CodecError("object arrays are not accepted")
            return arr
        if tag == "index":
            values = self.dec(n["v"], depth + 1)
            names = [self.dec(x, depth + 1) for x in n["names"]]
            if n["multi"]:
                return pd.MultiIndex.from_tuples(values, names=names)
            return pd.Index(values, name=names[0] if names else None)
        if tag == "set":
            return {self._key(self.dec(x, depth + 1)) for x in n["v"]}
        if tag in ("df", "series"):
            return self._frame(n, tag, depth)
        raise CodecError("unknown value in result")

    @staticmethod
    def _key(k: Any) -> Any:
        try:
            hash(k)
        except TypeError as e:
            raise CodecError("unhashable dict key") from e
        return k

    def _frame(self, n: dict, tag: str, depth: int):
        import pyarrow.parquet as pq
        body = pq.read_table(io.BytesIO(self._blob(n["b"]))).to_pandas()
        labels = [self.dec(x, depth + 1) for x in n["labels"]]
        cnames = [self.dec(x, depth + 1) for x in n["cnames"]]
        ix = n["index"]
        if ix["kind"] == "range":
            index = pd.RangeIndex(int(ix["start"]), int(ix["stop"]), int(ix["step"]),
                                  name=self.dec(ix["name"], depth + 1))
            data = body
        else:
            k = int(ix["n"])
            names = [self.dec(x, depth + 1) for x in ix["names"]]
            icols = [f"i{j}" for j in range(k)]
            index = (pd.MultiIndex.from_frame(body[icols], names=names) if k > 1
                     else pd.Index(body["i0"], name=names[0]))
            data = body.drop(columns=icols)
        data = data.copy()
        data.index = index
        data.columns = (pd.MultiIndex.from_tuples(labels, names=cnames) if n["multi"]
                        else pd.Index(labels, name=cnames[0] if cnames else None))
        if tag == "df":
            return data
        s = data.iloc[:, 0]
        s.name = self.dec(n["sname"], depth + 1)
        return s


def decode_payload(data: bytes) -> dict:
    """Parse bytes produced by encode_payload. Raises CodecError on anything
    that is not exactly our format (including a pickle). Never uses pickle."""
    try:
        if not isinstance(data, (bytes, bytearray)) or not data.startswith(MAGIC):
            raise CodecError("not a sandbox result (bad header)")
        pos = len(MAGIC)
        (hlen,) = struct.unpack_from("<I", data, pos); pos += 4
        header = json.loads(bytes(data[pos:pos + hlen]).decode("utf-8")); pos += hlen
        (nblobs,) = struct.unpack_from("<I", data, pos); pos += 4
        if nblobs > MAX_BLOBS:
            raise CodecError("too many binary parts")
        lens = struct.unpack_from(f"<{nblobs}Q", data, pos); pos += 8 * nblobs
        blobs = []
        for ln in lens:
            if pos + ln > len(data):
                raise CodecError("truncated result")
            blobs.append(bytes(data[pos:pos + ln])); pos += ln
        if pos != len(data):
            raise CodecError("trailing bytes in result")
        out = _Decoder(blobs).dec(header["payload"])
        if not isinstance(out, dict):
            raise CodecError("payload must be a dict")
        return out
    except CodecError:
        raise
    except (struct.error, KeyError, IndexError, TypeError, ValueError, UnicodeDecodeError,
            RecursionError, OSError) as e:
        raise CodecError(f"malformed sandbox result: {type(e).__name__}") from e
    except Exception as e:  # pyarrow / plotly parse errors
        raise CodecError(f"malformed sandbox result: {type(e).__name__}") from e
