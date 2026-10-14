"""
sandbox.py — Hardened code execution layer.

Two-layer defence:
  1. AST analysis  — catches imports, attribute access, dangerous calls
                     regardless of encoding tricks or obfuscation.
  2. Pattern guard — fast string scan as secondary check.
  3. exec() with a curated __builtins__ — a safe subset, not an empty
                     dict (see SAFE_BUILTINS below).

WHY NOT __builtins__: {} (as this used to be)
    Emptying __builtins__ entirely blocks EVERY builtin, not just
    dangerous ones — len(), round(), sum(), list(), range() all fail
    with NameError. That's not a security feature, it's collateral
    damage: it silently breaks completely ordinary pandas analysis code
    the LLM is very likely to generate (row counts via len(df), rounding
    a percentage, building a list of columns) with no security benefit,
    since none of those are attack surfaces. It even breaks the app's
    own documented prompt pattern in query_engine.py's
    build_analysis_prompt, which explicitly tells the LLM to write
    `if 'col' not in df.columns: raise ValueError('Column not found')`
    — ValueError itself is a builtin, and was unreachable under the old
    empty dict.

    SAFE_BUILTINS is an allowlist (same philosophy as sandbox_proxy.py):
    ordinary type/iteration/formatting functions and the common
    exception types, with everything actually dangerous (open, eval,
    exec, compile, input, __import__, getattr/setattr/delattr, vars,
    dir, globals, locals, breakpoint, memoryview, ...) left out. The
    AST/pattern layers above still independently block those names too
    — this is defense in depth, not a replacement for them.
"""

import ast
import builtins as _builtins_module

# ── Layer 2: string-level blocklist ──────────────────────────────────────────
BLOCKED_PATTERNS = [
    "while True", "while 1", "for(;;)",
    "open(", "eval(", "exec(",
    "os.", "sys.", "subprocess", "shutil",
    "socket", "requests", "urllib", "pathlib",
    "globals()", "locals()",
    "getattr", "setattr", "delattr",
]

BLOCKED_NAMES = {
    "open", "eval", "exec", "compile", "input",
    "__import__", "importlib", "breakpoint",
    "memoryview", "vars", "dir",
}

BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket",
    "requests", "urllib", "pathlib", "importlib",
    "ctypes", "pickle", "shelve", "multiprocessing",
    "threading", "signal", "pty", "fcntl",
}

# Ordinary, non-dangerous builtins that legitimate analysis code needs.
# Deliberately excludes anything in BLOCKED_NAMES above, plus anything
# else with file/process/introspection access (staticmethod/classmethod/
# super/object are left out too — never needed by generated analysis code,
# and no reason to widen the surface for something unused).
_SAFE_BUILTIN_NAMES = {
    # type conversion / construction
    "int", "float", "str", "bool", "list", "dict", "tuple", "set", "frozenset",
    # iteration / aggregation
    "len", "range", "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "sum", "min", "max", "any", "all", "abs", "round", "pow", "divmod",
    # read-only introspection — safe, no state access
    "isinstance", "issubclass", "type", "hasattr",
    # formatting
    "format", "repr", "ord", "chr", "hex", "oct", "bin",
    # exceptions — needed because query_engine.py's own prompt tells the
    # LLM to `raise ValueError(...)` for column-existence checks
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "ZeroDivisionError", "StopIteration", "RuntimeError",
    "ArithmeticError", "OverflowError", "NotImplementedError", "NameError",
}

SAFE_BUILTINS = {name: getattr(_builtins_module, name) for name in _SAFE_BUILTIN_NAMES}


# ── Layer 1: AST analysis ─────────────────────────────────────────────────────
def is_code_safe_ast(code: str) -> tuple[bool, str]:
    """
    Parse the code into an AST and walk every node.
    Catches imports, dangerous calls, and blocked module attribute access.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True, ""  # let exec() surface the syntax error

    for node in ast.walk(tree):
        # Block ALL import statements
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                return False, f"import '{name}' blocked"

        # Block calls to dangerous built-in names
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BLOCKED_NAMES:
                    return False, f"call to '{node.func.id}' blocked"
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id in BLOCKED_MODULES:
                        return False, f"module access '{node.func.value.id}.{node.func.attr}' blocked"

        # Defense in depth: attribute names that read/write files or evaluate
        # expressions are refused wherever they appear, even if some other
        # path ever hands generated code a raw pandas/plotly object.
        if isinstance(node, ast.Attribute) and node.attr in DENIED_ATTRS:
            return False, f"attribute '{node.attr}' blocked"

        # Block ANY private (underscore-prefixed) attribute access.
        # AI SECURITY — real escape found by running the evaluation harness
        # against real model output: the proxies in sandbox_proxy.py keep the
        # real object in a slot named `_obj` (`_mod`, `_con`, `_rel`, ...).
        # A slot is found by normal attribute lookup BEFORE __getattr__ runs,
        # so `df._obj` returned the raw, unproxied DataFrame and
        # `df._obj.to_csv("x.csv")` wrote a file from inside the sandbox.
        # The dunder check only catches `__x__`, not single-underscore names.
        # Nothing legitimate in analysis code needs a private attribute.
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return False, f"private attribute '{node.attr}' blocked"

        # Block attribute access on blocked modules
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in BLOCKED_MODULES:
                    return False, f"attribute access '{node.value.id}.{node.attr}' blocked"

        # Block Name references to dangerous identifiers
        if isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES:
                return False, f"name '{node.id}' blocked"

    return True, ""


DENIED_ATTRS = {
    "to_csv", "to_pickle", "to_excel", "to_json", "to_parquet", "to_hdf", "to_sql",
    "to_feather", "to_stata", "to_latex", "to_html", "to_markdown", "to_xml",
    "to_clipboard", "to_orc", "to_gbq", "to_xarray",
    "read_csv", "read_excel", "read_pickle", "read_json", "read_parquet", "read_sql",
    "read_html", "read_table", "read_fwf", "read_clipboard", "read_feather", "read_hdf",
    "savefig", "write_html", "write_image", "write_json", "write_kaleido",
    "open", "eval", "query", "style",
}


# ── Layer 2: string-level pattern guard ───────────────────────────────────────
# AI SECURITY / correctness — real false positive found on a user's dataset:
# the guard used plain substring matching on the WHOLE source, so a column
# named `total_of_special_requests` (inside a string literal, df['...']) tripped
# the "requests" pattern and the cleaning agent's fix code was rejected. The
# same flaw hit any column containing "socket", "pathlib", "...pos." ("os."),
# "reopen(" ("open("), and so on. Fix:
#   1. string literals and comments are blanked out before scanning (data/column
#      names live there; code that EXECUTES a string needs eval/exec/getattr,
#      which the AST layer and the name checks below still block), and
#   2. identifiers are matched as whole words, so `total_of_special_requests`
#      or `cos.` can never match `requests` / `os.`.
# f-strings are still scanned, since their {...} parts are real code.
import io as _io
import re as _re
import tokenize as _tokenize

_WORD_PATTERNS = [
    ("open(", r"\bopen\s*\("), ("eval(", r"\beval\s*\("), ("exec(", r"\bexec\s*\("),
    ("os.", r"\bos\s*\."), ("sys.", r"\bsys\s*\."),
    ("subprocess", r"\bsubprocess\b"), ("shutil", r"\bshutil\b"),
    ("socket", r"\bsocket\b"), ("requests", r"\brequests\b"),
    ("urllib", r"\burllib\b"), ("pathlib", r"\bpathlib\b"),
    ("globals()", r"\bglobals\s*\(\s*\)"), ("locals()", r"\blocals\s*\(\s*\)"),
    ("getattr", r"\bgetattr\b"), ("setattr", r"\bsetattr\b"), ("delattr", r"\bdelattr\b"),
]
_LOOP_PATTERNS = ["while True", "while 1", "for(;;)"]


def _strip_literals(code: str) -> str:
    """Blank out plain string literals and comments (keeping line structure).
    f-strings are left intact. On any tokenizer error, returns code unchanged
    (stricter, never looser)."""
    try:
        toks = list(_tokenize.generate_tokens(_io.StringIO(code).readline))
    except (_tokenize.TokenError, IndentationError, SyntaxError):
        return code
    lines = code.splitlines(keepends=True)
    # collect spans to blank, apply from the end so offsets stay valid
    spans = []
    for t in toks:
        if t.type == _tokenize.COMMENT:
            spans.append((t.start, t.end))
        elif t.type == _tokenize.STRING:
            prefix = _re.match(r"[A-Za-z]*", t.string).group(0).lower()
            if "f" not in prefix:
                spans.append((t.start, t.end))
    for (sr, sc), (er, ec) in sorted(spans, reverse=True):
        if sr == er:
            ln = lines[sr - 1]
            lines[sr - 1] = ln[:sc] + " " * (ec - sc) + ln[ec:]
        else:  # multi-line string: blank each covered line segment
            lines[sr - 1] = lines[sr - 1][:sc] + " " * (len(lines[sr - 1]) - sc - 1) + "\n"
            for r in range(sr, er - 1):
                lines[r] = " " * (len(lines[r]) - 1) + "\n"
            lines[er - 1] = " " * ec + lines[er - 1][ec:]
    return "".join(lines)


def is_code_safe_patterns(code: str) -> tuple[bool, str]:
    # Dunders stay blocked everywhere, strings included — nothing legitimate
    # needs `__`, and it is the classic introspection route.
    if "__" in code:
        return False, "__ (dunder) usage blocked"
    scan = _strip_literals(code)
    for pattern in _LOOP_PATTERNS:
        if pattern in scan:
            return False, f"pattern '{pattern}' blocked"
    for label, rx in _WORD_PATTERNS:
        if _re.search(rx, scan):
            return False, f"pattern '{label}' blocked"
    return True, ""


# ── Combined guard ────────────────────────────────────────────────────────────
def is_code_safe(code: str) -> tuple[bool, str]:
    """Run both layers. Both must pass."""
    safe, reason = is_code_safe_ast(code)
    if not safe:
        return False, reason
    return is_code_safe_patterns(code)


# ── Hardened exec ─────────────────────────────────────────────────────────────
def safe_exec(code: str, local_vars: dict) -> tuple[bool, str]:
    """
    Execute code in a hardened sandbox.
    - Runs both safety layers before exec.
    - Passes SAFE_BUILTINS (a curated allowlist) instead of the full
      builtins module — ordinary functions work, dangerous ones don't
      exist in the namespace at all.
    """
    safe, reason = is_code_safe(code)
    if not safe:
        return False, f"Security block: {reason}"
    try:
        exec(code, {"__builtins__": SAFE_BUILTINS}, local_vars)
        return True, ""
    except Exception as e:
        return False, str(e)
