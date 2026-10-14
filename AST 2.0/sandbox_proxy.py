"""
sandbox_proxy.py — Capability-limited proxies for pandas, numpy, matplotlib,
and plotly, used inside the sandboxed exec() namespace instead of the raw
modules.

WHY THIS EXISTS
    sandbox.py's AST/pattern guard blocks imports, dunders, and calls to a
    fixed list of dangerous *names* (open, eval, exec, etc). It does NOT
    restrict what generated code can do with modules that are already
    handed to it in the exec namespace — pd, np, plt, px, go. Those modules
    expose file I/O (pd.read_pickle, pd.read_csv, plt.savefig,
    fig.write_html) and even arbitrary-code-execution surfaces
    (pd.read_pickle unpickles arbitrary objects) that the AST guard has no
    way to catch, because calling `pd.read_pickle(...)` is just a normal
    attribute-call on a name that was never blocked.

HOW IT WORKS
    Allowlist, not blocklist. Each proxy only forwards attribute access for
    names explicitly listed below. Anything else raises AttributeError —
    the same failure mode the app's existing retry-on-error loop already
    handles, so no other code needs to change to benefit from this.

WHAT THIS IS NOT
    This is one layer of defense, not a replacement for process/resource
    isolation. It stops a query from reading/writing arbitrary files or
    deserializing a malicious pickle. It does not limit CPU time or memory
    — pair it with the exec timeout / resource-cap layer for that.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = go = None

# AI SECURITY — see _wrap()'s docstring below for the full story: these
# are the intermediate object types df.groupby()/.rolling()/.expanding()
# return, which _wrap() must ALSO proxy (a real sandbox-escape bug found
# by test_sandbox_security.py, fixed here). Prefer the public
# pandas.api.typing namespace (stable since pandas 2.1); fall back to the
# private pandas.core.* path for pandas 2.0.x, which predates it — same
# defensive-import pattern already used above for optional plt/px/go.
try:
    from pandas.api.typing import DataFrameGroupBy, SeriesGroupBy, Rolling, Expanding
except ImportError:
    from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
    from pandas.core.window import Rolling, Expanding

# Anything reachable through an ALLOWED_FRAME_METHODS call
# (groupby/rolling/expanding) that is NOT itself a terminal DataFrame/
# Series result. Every one of these needs the exact same proxying as a
# DataFrame/Series — see _wrap().
_CHAINABLE_TYPES = (DataFrameGroupBy, SeriesGroupBy, Rolling, Expanding)
try:  # resample()/ewm() return these; they need the same proxying as groupby
    from pandas.core.resample import Resampler as _Resampler
    from pandas.core.window import ExponentialMovingWindow as _EWM
    _CHAINABLE_TYPES = _CHAINABLE_TYPES + (_Resampler, _EWM)
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────
# ALLOWLISTS
# ─────────────────────────────────────────────────────────────────────────

# Methods safe to call on a DataFrame or Series.
# Deliberately EXCLUDES: to_csv, to_pickle, to_sql, to_excel, to_hdf,
# to_parquet, to_feather, to_clipboard (all disk/OS I/O), and eval/query
# (pandas expression evaluation — an unnecessary extra surface).
ALLOWED_FRAME_METHODS = {
    "head", "tail", "sample", "copy", "describe", "info",
    "sort_values", "sort_index", "nlargest", "nsmallest",
    "groupby", "agg", "aggregate", "apply", "applymap", "map",
    "merge", "join", "pivot", "pivot_table", "melt",
    "sum", "mean", "median", "std", "var", "min", "max", "count",
    "nunique", "unique", "value_counts", "mode",
    "corr", "cov", "quantile", "cumsum", "cumprod", "cummax", "cummin",
    "diff", "pct_change", "rolling", "expanding",
    "isnull", "isna", "notnull", "notna", "fillna", "dropna",
    "interpolate", "ffill", "bfill",  # added for the cleaning agent — legitimate
                                        # strategies for time-ordered / sequential data,
                                        # no file I/O, same allowlist philosophy as fillna
    "drop", "drop_duplicates", "duplicated", "rename",
    "reset_index", "set_index", "astype", "round", "abs", "clip", "replace",
    "select_dtypes", "filter", "assign",
    "to_dict", "to_string", "to_numpy", "to_frame", "to_list", "tolist",
    "plot",  # matplotlib accessor — returns an Axes, no file I/O itself
    # Added after the evaluation harness showed real model output failing on
    # these. All are pure in-memory computation, no I/O, no expression eval:
    "idxmax", "idxmin",          # "which region has the highest X" — the single most common idiom
    "isin", "between",           # row filtering
    "shift", "rank",             # time-series / ranking
    "where", "mask",             # conditional replacement
    "any", "all",                # boolean reductions
    "transform",                 # group-wise transforms; lambdas are re-proxied (see _wrap_callable)
    "size", "first", "last", "cumcount", "prod", "sem", "skew",  # groupby.size() etc. — pure in-memory
    # Broad pass: every entry below is pure in-memory computation. Anything that
    # touches files, the OS, expression strings (eval/query), Styler, or yields
    # raw un-proxied rows (iterrows/items/itertuples) is deliberately absent.
    "resample", "ewm", "asfreq", "to_period", "to_timestamp",
    "ngroup", "nth", "get_group", "kurt", "kurtosis", "cumcount",
    "argmax", "argmin", "squeeze", "explode", "stack", "unstack", "swaplevel",
    "reindex", "truncate", "add_prefix", "add_suffix", "combine_first",
    "equals", "memory_usage", "droplevel", "rename_axis", "infer_objects",
    "convert_dtypes", "first_valid_index", "last_valid_index", "dot",
    "eq", "ne", "lt", "le", "gt", "ge",
    "add", "sub", "mul", "div", "truediv", "floordiv", "mod", "pow",
    "radd", "rsub", "rmul", "rdiv", "rtruediv", "rfloordiv", "rmod", "rpow",
    "product", "mad", "ptp", "idxmax", "idxmin", "autocorr", "nlargest",
    "factorize", "searchsorted", "compare", "clip", "round",
    "get", "keys", "agg", "median", "prod",
}

# Label/position indexers (df.loc / df.iloc / df.at / df.iat). These are NOT
# returned raw: a raw indexer exposes `.obj`, which is the unproxied frame.
# _IndexerProxy exposes only item get/set.
INDEXER_ATTRS = {"loc", "iloc", "at", "iat"}

# Attributes (not methods) safe to read directly.
ALLOWED_FRAME_ATTRS = {
    "columns", "dtypes", "shape", "index", "values", "empty", "T",
    "str", "dt", "cat",  # pandas accessor namespaces
}

# Top-level pandas functions safe to call.
# Excludes ALL read_*/to_* I/O, ExcelWriter, HDFStore, read_sql, eval.
ALLOWED_PD_FUNCS = {
    "DataFrame", "Series", "concat", "merge", "pivot_table", "melt",
    "to_datetime", "to_numeric", "to_timedelta", "date_range",
    "isna", "isnull", "notna", "notnull", "cut", "qcut", "get_dummies",
    "NaT", "NA", "Timestamp", "Timedelta",
    "crosstab", "factorize", "unique", "period_range", "timedelta_range",
    "Period", "DateOffset", "Categorical", "CategoricalDtype", "Index",
    "DatetimeIndex", "MultiIndex", "IntervalIndex", "Interval", "interval_range",
    "wide_to_long", "pivot", "value_counts", "eval_placeholder_removed",
}
ALLOWED_PD_FUNCS.discard("eval_placeholder_removed")

# Top-level numpy functions safe to call.
# Excludes np.load/save/fromfile/tofile/loadtxt/savetxt (file I/O) and
# np.lib/ctypeslib entry points.
ALLOWED_NP_FUNCS = {
    "array", "arange", "linspace", "zeros", "ones", "full",
    "mean", "median", "std", "var", "sum", "min", "max", "abs",
    "round", "sqrt", "log", "log2", "log10", "exp", "power",
    "where", "select", "unique", "sort", "argsort", "percentile",
    "corrcoef", "cov", "nan", "inf", "pi", "e",
    "isnan", "isinf", "isfinite", "concatenate", "reshape",
    "cumsum", "cumprod", "diff", "clip", "isin", "nanmean", "nansum", "nanmedian",
    "nanstd", "nanmax", "nanmin", "floor", "ceil", "trunc", "sin", "cos", "tan",
    "arctan", "arctan2", "log1p", "expm1", "sign", "maximum", "minimum",
    "argmax", "argmin", "histogram", "digitize", "bincount", "zeros_like",
    "ones_like", "tile", "repeat", "stack", "vstack", "hstack", "meshgrid",
    "mod", "average", "ptp", "quantile", "any", "all", "count_nonzero",
    "searchsorted", "flip", "isclose", "int64", "float64", "int32", "float32",
    "datetime64", "timedelta64", "bool_", "nanpercentile", "interp", "polyfit",
    "polyval", "cbrt", "square", "hypot", "radians", "degrees", "tanh",
    "empty", "full_like", "identity", "eye", "outer", "dot",
}

# matplotlib.pyplot functions safe to call.
# Excludes savefig, show (already stripped by extract_and_clean_code,
# blocked again here as defense in depth), imsave, imread (file I/O).
ALLOWED_PLT_FUNCS = {
    "figure", "subplots", "plot", "bar", "barh", "hist", "scatter",
    "boxplot", "pie", "imshow", "xlabel", "ylabel", "title", "legend",
    "xticks", "yticks", "grid", "tight_layout", "gcf", "gca", "close",
    "subplots_adjust", "axhline", "axvline", "fill_between",
}

# plotly.express chart constructors — matches exactly what query_engine.py's
# prompt instructs the LLM to use.
ALLOWED_PX_FUNCS = {
    "bar", "line", "scatter", "pie", "histogram", "imshow",
    "box", "area", "scatter_3d",
    "violin", "strip", "density_heatmap", "density_contour", "treemap",
    "sunburst", "funnel", "scatter_matrix", "bar_polar", "line_polar",
    "scatter_polar", "line_3d", "ecdf", "parallel_coordinates", "timeline",
    "pie", "icicle", "funnel_area",
}

ALLOWED_GO_ATTRS = {
    "Figure", "Scatter", "Bar", "Pie", "Histogram", "Box",
    "Heatmap", "Violin", "Scatter3d", "Funnel", "Waterfall", "Treemap",
    "Sunburst", "Indicator", "Table", "Contour", "Surface", "Candlestick",
    "Scatterpolar", "Barpolar", "Layout", "Histogram2d", "Sankey",
}

# Instance methods on a returned Figure (plotly or matplotlib) that write to
# disk or open an external viewer — blocked even though the Figure itself
# is returned to the caller as `result`.
BLOCKED_FIGURE_METHODS = {
    "write_html", "write_image", "write_json", "write_kaleido",
    "show", "savefig",
}


# ─────────────────────────────────────────────────────────────────────────
# WRAPPING
# ─────────────────────────────────────────────────────────────────────────

def _is_figure(obj) -> bool:
    if plt is not None and hasattr(obj, "savefig") and hasattr(obj, "gca"):
        return True
    if px is not None and "plotly.graph_objs" in str(type(obj)):
        return True
    return False


def _wrap(obj):
    """Wrap a value returned from an allowed call, if it needs gating.

    AI SECURITY — real bug found and fixed here (via
    test_sandbox_security.py): this used to check ONLY
    isinstance(obj, (pd.DataFrame, pd.Series)). But df.groupby(...) —
    itself an ALLOWED_FRAME_METHODS entry — returns a DataFrameGroupBy,
    not a DataFrame. That object was never a DataFrame/Series, so it
    fell through this check entirely and was returned to generated code
    completely UNWRAPPED — meaning `df.groupby('x').sum().to_pickle(...)`
    (or .to_csv, or anything else off the allowlist) executed with ZERO
    restriction the instant a .groupby()/.rolling()/.expanding() call
    was in the chain, silently bypassing the entire proxy system for
    what is likely the single most common pattern in this app's
    LLM-generated code. _CHAINABLE_TYPES (defined above) closes this —
    every intermediate object type these three allowed methods can
    return is now wrapped exactly like a DataFrame/Series, using the
    SAME _FrameProxy class and the SAME allowlist, since the same
    methods (sum/mean/agg/apply/...) are the legitimate operations on
    both a DataFrame and a GroupBy/Rolling/Expanding object."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return _FrameProxy(obj)
    if isinstance(obj, _CHAINABLE_TYPES):
        return _FrameProxy(obj)
    if _is_figure(obj):
        return _FigureProxy(obj)
    return obj


# AI SECURITY — second sandbox escape found while auditing test coverage:
# `df.apply(lambda col: col.to_csv("x.csv"))` and `.agg(lambda s: ...)` wrote
# files. pandas calls the user's lambda with its OWN raw Series/DataFrame, not
# the proxy, so everything off the allowlist was reachable inside the lambda.
# Fix: every callable handed to a proxied call is wrapped so the pandas objects
# pandas passes into it are re-proxied (and anything it returns is unwrapped
# again). Containers are handled recursively (agg({"col": lambda s: ...})).

def _unwrap_value(v):
    return v._obj if isinstance(v, (_FrameProxy, _FigureProxy)) else v


def _wrap_callable(fn):
    def _inner(*a, **k):
        a = tuple(_wrap(x) for x in a)
        k = {kk: _wrap(vv) for kk, vv in k.items()}
        return _unwrap_one(fn(*a, **k))
    return _inner


def _unwrap_one(v):
    if isinstance(v, (_FrameProxy, _FigureProxy)):
        return v._obj
    if type(v) in (list, tuple):
        return type(v)(_unwrap_one(x) for x in v)
    if type(v) is dict:
        return {kk: _unwrap_one(x) for kk, x in v.items()}
    if callable(v) and not isinstance(v, type):
        return _wrap_callable(v)
    return v


def _unwrap_args(args, kwargs):
    args = tuple(_unwrap_one(a) for a in args)
    kwargs = {k: _unwrap_one(v) for k, v in kwargs.items()}
    return args, kwargs


def _unwrap_index_key(key):
    """Unwraps proxies inside an indexer key. Callable keys are refused:
    `df.loc[lambda d: ...]` would hand the lambda the RAW frame."""
    if isinstance(key, (_FrameProxy, _FigureProxy)):
        return key._obj
    if isinstance(key, tuple):
        return tuple(_unwrap_index_key(k) for k in key)
    if isinstance(key, slice):
        return slice(_unwrap_index_key(key.start), _unwrap_index_key(key.stop), _unwrap_index_key(key.step))
    if isinstance(key, list):
        return [_unwrap_index_key(k) for k in key]
    if callable(key):
        raise TypeError("callable keys are not allowed in .loc/.iloc inside the sandbox")
    return key


class _IndexerProxy:
    """Wraps df.loc / df.iloc / df.at / df.iat. Only `indexer[key]` and
    `indexer[key] = value` exist; there is deliberately no attribute
    passthrough, so `df.loc.obj` (the raw frame) is unreachable."""

    __slots__ = ("_idx",)

    def __init__(self, idx):
        object.__setattr__(self, "_idx", idx)

    def __getitem__(self, key):
        return _wrap(object.__getattribute__(self, "_idx")[_unwrap_index_key(key)])

    def __setitem__(self, key, value):
        if isinstance(value, (_FrameProxy, _FigureProxy)):
            value = value._obj
        object.__getattribute__(self, "_idx")[_unwrap_index_key(key)] = value

    def __setattr__(self, name, value):
        raise AttributeError("assigning attributes is not permitted in the sandbox")


class _FrameProxy:
    """Wraps a DataFrame/Series — or a GroupBy/Rolling/Expanding object
    from an in-progress .groupby()/.rolling()/.expanding() chain, see
    _CHAINABLE_TYPES and _wrap()'s docstring — so only allowlisted
    methods/attrs are reachable regardless of which of those the
    underlying object actually is."""

    __slots__ = ("_obj",)

    def __init__(self, obj):
        object.__setattr__(self, "_obj", obj)

    def __getattr__(self, name):
        obj = object.__getattribute__(self, "_obj")

        if name in ALLOWED_FRAME_ATTRS:
            val = getattr(obj, name)
            return _wrap(val)

        if name in INDEXER_ATTRS and isinstance(obj, (pd.DataFrame, pd.Series)):
            return _IndexerProxy(getattr(obj, name))

        # df.size / series.size is a plain int attribute; groupby.size() is a method.
        if name == "size" and isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.size

        if name in ALLOWED_FRAME_METHODS:
            attr = getattr(obj, name)

            def _call(*args, **kwargs):
                args, kwargs = _unwrap_args(args, kwargs)
                return _wrap(attr(*args, **kwargs))

            return _call

        raise AttributeError(
            f"'{name}' is blocked in the sandbox for security "
            f"(file I/O, expression evaluation, and similar operations are disabled)"
        )

    def __setattr__(self, name, value):
        raise AttributeError("assigning attributes on df is not permitted in the sandbox")

    def __len__(self):
        return len(object.__getattribute__(self, "_obj"))

    def __getitem__(self, key):
        obj = object.__getattribute__(self, "_obj")
        if isinstance(key, _FrameProxy):
            key = object.__getattribute__(key, "_obj")
        return _wrap(obj[key])

    def __setitem__(self, key, value):
        obj = object.__getattribute__(self, "_obj")
        if isinstance(value, (_FrameProxy, _FigureProxy)):
            value = value._obj
        obj[key] = value

    def __repr__(self):
        return repr(object.__getattribute__(self, "_obj"))

    def __iter__(self):
        return iter(object.__getattribute__(self, "_obj"))

    def __contains__(self, item):
        return item in object.__getattribute__(self, "_obj")


# ── Operators ────────────────────────────────────────────────────────────────
# AI SECURITY / FUNCTIONALITY — found by the evaluation harness: _FrameProxy
# defined NO operators at all, so `df['rev'] > 15`, `df['a'] * 2`,
# `df['a'] + df['b']`, `~mask` and `(m1) & (m2)` all raised TypeError, and
# `df['region'] == 'S'` silently evaluated to plain False (then crashed as
# df[False]). Every filter and every derived column failed. Python looks
# operators up on the TYPE, so __getattr__ never saw them. They are pure
# element-wise computation on the underlying object; the result is re-wrapped
# so the proxy is never lost, and proxy operands are unwrapped first.

def _unwrap_operand(v):
    return object.__getattribute__(v, "_obj") if isinstance(v, _FrameProxy) else v


def _make_binary_op(op_name):
    def _op(self, other):
        obj = object.__getattribute__(self, "_obj")
        return _wrap(getattr(obj, op_name)(_unwrap_operand(other)))
    _op.__name__ = op_name
    return _op


def _make_unary_op(op_name):
    def _op(self):
        return _wrap(getattr(object.__getattribute__(self, "_obj"), op_name)())
    _op.__name__ = op_name
    return _op


for _n in ("lt", "le", "gt", "ge", "eq", "ne",
           "add", "sub", "mul", "truediv", "floordiv", "mod", "pow",
           "radd", "rsub", "rmul", "rtruediv", "rfloordiv", "rmod", "rpow",
           "and", "or", "xor", "rand", "ror", "rxor"):
    setattr(_FrameProxy, f"__{_n}__", _make_binary_op(f"__{_n}__"))
for _n in ("neg", "pos", "abs", "invert"):
    setattr(_FrameProxy, f"__{_n}__", _make_unary_op(f"__{_n}__"))
# Defining __eq__ removes the default __hash__; proxies are never dict keys.
_FrameProxy.__hash__ = None


class _FigureProxy:
    """Wraps a plotly/matplotlib Figure. Blocks only the disk-writing /
    display methods — everything else passes straight through, since a
    Figure is normally a terminal output (assigned to `result`), not
    chained into further sandboxed calls."""

    __slots__ = ("_obj",)

    def __init__(self, obj):
        object.__setattr__(self, "_obj", obj)

    def __getattr__(self, name):
        if name in BLOCKED_FIGURE_METHODS:
            raise AttributeError(
                f"'{name}' is blocked in the sandbox for security "
                f"(figures cannot be written to disk or displayed externally)"
            )
        return getattr(object.__getattribute__(self, "_obj"), name)

    def __repr__(self):
        return repr(object.__getattribute__(self, "_obj"))


class _ModuleProxy:
    """Generic allowlist-gated wrapper around a module (pd, np, plt, px, go)."""

    __slots__ = ("_mod", "_allowed")

    def __init__(self, mod, allowed):
        object.__setattr__(self, "_mod", mod)
        object.__setattr__(self, "_allowed", allowed)

    def __getattr__(self, name):
        allowed = object.__getattribute__(self, "_allowed")
        mod = object.__getattribute__(self, "_mod")

        if name not in allowed:
            raise AttributeError(
                f"'{name}' is blocked in the sandbox for security "
                f"(only a fixed set of analysis/charting functions is available)"
            )

        attr = getattr(mod, name)
        if not callable(attr):
            return attr  # constants: np.nan, np.pi, pd.NaT, ...

        def _call(*args, **kwargs):
            args, kwargs = _unwrap_args(args, kwargs)
            return _wrap(attr(*args, **kwargs))

        return _call

    def __setattr__(self, name, value):
        raise AttributeError("assigning module attributes is not permitted in the sandbox")


# ─────────────────────────────────────────────────────────────────────────
# MULTI-TABLE SQL SUPPORT (feature: joining/querying several uploaded
# datasets at once, DuckDB-backed)
# ─────────────────────────────────────────────────────────────────────────

class _DuckDBRelationProxy:
    """Wraps the result of SafeDuckDBConnection.sql() — exposes ONLY
    .df(), converting the query result into an already-restricted
    _FrameProxy DataFrame. Every other DuckDBPyRelation method
    (.write_csv, .to_parquet, .to_table, ...) is unreachable — the only
    sanctioned way out of a SQL query result in this sandbox is a
    pandas DataFrame, exactly like every other result type here.
    """

    __slots__ = ("_rel",)

    def __init__(self, rel):
        object.__setattr__(self, "_rel", rel)

    def df(self):
        rel = object.__getattribute__(self, "_rel")
        return _wrap(rel.df())

    def __getattr__(self, name):
        raise AttributeError(
            f"'{name}' is blocked in the sandbox — only .df() is exposed on a query result"
        )

    def __setattr__(self, name, value):
        raise AttributeError("assigning attributes on the sandboxed query result is not permitted")


class SafeDuckDBConnection:
    """Wraps a pre-configured DuckDB connection — external file/network
    access disabled via enable_external_access=False, tables already
    registered by trusted orchestration code BEFORE this wrapper is
    built — exposing ONLY .sql(), the single operation multi-table
    query code needs.

    Every other DuckDBPyConnection method is deliberately unreachable:
      - .register() isn't re-exposed here: it already ran (on the raw,
        unwrapped connection) before generated code gets a namespace at
        all. Re-exposing it would let generated code register its own
        arbitrary tables — not a large risk on its own, but there's no
        reason to widen the surface for something this feature never
        needs.
      - .connect() would be far more serious: it creates a BRAND NEW
        connection with none of this wrapper's restrictions, which
        would completely defeat the enable_external_access lockdown
        that keeps SQL text from reading arbitrary files or making
        network calls via DuckDB's own read_csv/httpfs functions (see
        multi_table_service.py's docstring for why that lockdown, not a
        SQL-keyword denylist, is the real safety boundary here).
    """

    __slots__ = ("_con",)

    def __init__(self, con):
        object.__setattr__(self, "_con", con)

    def sql(self, query: str):
        con = object.__getattribute__(self, "_con")
        return _DuckDBRelationProxy(con.sql(query))

    def __getattr__(self, name):
        raise AttributeError(
            f"'{name}' is blocked in the sandbox — only .sql() is exposed on this connection"
        )

    def __setattr__(self, name, value):
        raise AttributeError("assigning attributes on the sandboxed connection is not permitted")


# ─────────────────────────────────────────────────────────────────────────
# PUBLIC API — this is what the rest of the app calls
# ─────────────────────────────────────────────────────────────────────────

def build_safe_namespace(df: pd.DataFrame, include_charts: bool = True) -> dict:
    """
    Build the exec() local namespace for one sandboxed run.
    Drop-in replacement for the raw {"df": df, "pd": pd, "np": np, ...}
    dict previously passed to safe_exec().
    """
    ns = {
        "df": _FrameProxy(df),
        "pd": _ModuleProxy(pd, ALLOWED_PD_FUNCS),
        "np": _ModuleProxy(np, ALLOWED_NP_FUNCS),
    }
    if include_charts:
        if plt is not None:
            ns["plt"] = _ModuleProxy(plt, ALLOWED_PLT_FUNCS)
        if px is not None:
            ns["px"] = _ModuleProxy(px, ALLOWED_PX_FUNCS)
            ns["go"] = _ModuleProxy(go, ALLOWED_GO_ATTRS)
    return ns


def build_safe_namespace_multi(con, include_charts: bool = False) -> dict:
    """
    Exec namespace for the multi-table SQL feature — built around a
    single pre-configured, safety-locked DuckDB connection (tables
    already registered by trusted orchestration code) rather than a
    single `df`. Deliberately does NOT expose the named DataFrames as
    Python variables at all — the only sanctioned way to touch the data
    is through SQL via `con.sql(...)`, matching what this feature is
    actually for. pd/np stay available for light post-processing of the
    query result (renaming a column, rounding a value) before
    assignment to `result`.
    """
    ns = {
        "con": SafeDuckDBConnection(con),
        "pd": _ModuleProxy(pd, ALLOWED_PD_FUNCS),
        "np": _ModuleProxy(np, ALLOWED_NP_FUNCS),
    }
    if include_charts:
        if plt is not None:
            ns["plt"] = _ModuleProxy(plt, ALLOWED_PLT_FUNCS)
        if px is not None:
            ns["px"] = _ModuleProxy(px, ALLOWED_PX_FUNCS)
            ns["go"] = _ModuleProxy(go, ALLOWED_GO_ATTRS)
    return ns


def unwrap_result(value):
    """
    Call this on anything pulled out of local_vars after exec() returns
    (local_vars['result'], local_vars['cleaned_df'], etc.) before handing
    it to the rest of the app. ui.py / st.dataframe / st.plotly_chart all
    expect the real pandas/plotly object, not a proxy.
    """
    if isinstance(value, (_FrameProxy, _FigureProxy)):
        return object.__getattribute__(value, "_obj")
    return value
