"""
Functional proof for the SAFE_BUILTINS fix.

Confirms:
  1. Ordinary builtins the LLM is very likely to generate (len, round,
     sum, list, range, the ValueError pattern query_engine.py's own
     prompt asks for) now work, where they previously failed with
     "name 'X' is not defined" under __builtins__: {}.
  2. Every dangerous builtin (open, eval, exec, compile, __import__,
     getattr/setattr/delattr, vars, dir) is STILL blocked — this fix
     widens the allowlist, it does not loosen the AST/pattern guards.

Run: python3 test_safe_builtins.py
"""

from sandbox import safe_exec

print("=" * 70)
print("PART 1 — ordinary builtins, previously broken, should now WORK")
print("=" * 70)

legit = [
    ("len() on a list",        "result = len([1, 2, 3, 4])"),
    ("round() a float",        "result = round(3.14159, 2)"),
    ("sum() a list",           "result = sum([10, 20, 30])"),
    ("list(range(...))",       "result = list(range(5))"),
    ("min/max/abs",            "result = (min(3, 1), max(3, 1), abs(-7))"),
    ("sorted() with key",      "result = sorted([3, 1, 2])"),
    ("enumerate()",            "result = list(enumerate(['a', 'b']))"),
    ("the exact pattern query_engine.py's own prompt asks the LLM to write",
     "cols = ['x', 'y']\nif 'x' not in cols: raise ValueError('Column not found')\nresult = 'ok'"),
]

for label, code in legit:
    success, error = safe_exec(code, {})
    status = "OK" if success else "!!! BROKEN !!!"
    print(f"\n[{status}] {label}")
    print(f"    code: {code!r}")
    if not success:
        print(f"    error: {error}")

print("\n" + "=" * 70)
print("PART 2 — dangerous builtins, should STILL be blocked (regression check)")
print("=" * 70)

dangerous = [
    ("open()",       "result = open('/etc/passwd').read()"),
    ("eval()",       "result = eval('1+1')"),
    ("exec()",       "exec('x=1')\nresult = 1"),
    ("compile()",    "result = compile('1+1', '<s>', 'eval')"),
    ("__import__()", "result = __import__('os')"),
    ("getattr()",    "result = getattr([], 'append')"),
    ("vars()",       "result = vars()"),
    ("dir()",        "result = dir()"),
]

for label, code in dangerous:
    success, error = safe_exec(code, {})
    status = "BLOCKED (correct)" if not success else "!!! NOT BLOCKED !!!"
    print(f"\n[{status}] {label}")
    print(f"    error: {error}")

print("\n" + "=" * 70)
print("Done.")
