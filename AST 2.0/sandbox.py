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


# ── Layer 2: string-level pattern guard ───────────────────────────────────────
def is_code_safe_patterns(code: str) -> tuple[bool, str]:
    if "__" in code:
        return False, "__ (dunder) usage blocked"
    for pattern in BLOCKED_PATTERNS:
        if pattern in code:
            return False, f"pattern '{pattern}' blocked"
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
