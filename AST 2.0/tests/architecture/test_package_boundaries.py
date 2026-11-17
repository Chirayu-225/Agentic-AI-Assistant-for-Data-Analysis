"""
test_package_boundaries.py - the packages keep a one-way layering.

The backend is split into packages with one job each. This test reads every import in
every module (including imports written inside functions) and compares the package-to-
package graph with the map below. Run: python -m tests.architecture.test_package_boundaries

It fails when
  1. a package imports another package that the map does not allow (a new coupling),
  2. a map entry no longer exists in the code (the map must stay true),
  3. a cycle appears between packages, other than the one accepted cycle named below.

Layers, lowest first (a package may use the ones above it in this list, never below):
  audit, llm, storage            leaf packages: import no other package
  security, execution            use audit only
  ingestion <-> data_cleaning    one ACCEPTED cycle: the file reader applies the cleaning
                                 package's missing-value rules, and the cleaning agent reads
                                 files. Splitting it means moving the NA-token and stage-0
                                 readers into ingestion; recorded as future work.
  analysis                       question answering; uses everything below it
  rag                            document questions; uses ingestion, llm, security
  evaluation                     measures the others; nothing imports it
  api.py                         the HTTP layer; nothing imports it
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PACKAGES = ["analysis", "audit", "data_cleaning", "evaluation", "execution",
            "ingestion", "llm", "rag", "security", "storage"]

ALLOWED = {
    "audit": set(),
    "llm": set(),
    "storage": set(),
    "security": {"audit"},
    "execution": {"audit"},
    "ingestion": {"data_cleaning", "storage"},
    "data_cleaning": {"audit", "execution", "ingestion", "llm", "security", "storage"},
    "analysis": {"audit", "data_cleaning", "execution", "ingestion", "llm", "security", "storage"},
    "rag": {"ingestion", "llm", "security"},
    "evaluation": {"analysis", "data_cleaning", "execution", "ingestion", "rag"},
    "api": {"analysis", "audit", "data_cleaning", "execution", "ingestion", "llm", "rag",
            "security", "storage"},
}

ACCEPTED_CYCLES = [{"ingestion", "data_cleaning"}]

FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" - {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def modules():
    """(owner package or 'api', relative path) for every backend module."""
    out = []
    for pkg in PACKAGES:
        for dirpath, dirs, files in os.walk(os.path.join(ROOT, pkg)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith(".py"):
                    out.append((pkg, os.path.relpath(os.path.join(dirpath, f), ROOT)))
    out.append(("api", "api.py"))
    return sorted(out)


def imported_packages(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    found = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            names = [n.module]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top in PACKAGES:
                found.add(top)
    return found


def strongly_connected(graph):
    """Tarjan: groups of packages that can all reach each other (size > 1 = a cycle)."""
    index, low, on, stack, out, counter = {}, {}, set(), [], [], [0]

    def visit(v):
        index[v] = low[v] = counter[0]; counter[0] += 1
        stack.append(v); on.add(v)
        for w in graph.get(v, ()):
            if w not in index:
                visit(w); low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = set()
            while True:
                w = stack.pop(); on.discard(w); comp.add(w)
                if w == v:
                    break
            out.append(comp)
    for v in list(graph):
        if v not in index:
            visit(v)
    return [c for c in out if len(c) > 1]


def main():
    edges = {p: set() for p in ALLOWED}
    where = {}
    for owner, path in modules():
        for dep in imported_packages(path):
            if dep != owner:
                edges[owner].add(dep)
                where.setdefault((owner, dep), []).append(path)

    print("=" * 70 + "\nPART 1 - every import is allowed by the map\n" + "=" * 70)
    for owner in sorted(ALLOWED):
        extra = sorted(edges[owner] - ALLOWED[owner])
        check(f"{owner}: no unlisted package imports", not extra,
              "; ".join(f"{d} (in {', '.join(where[(owner, d)][:3])})" for d in extra))

    print("\n" + "=" * 70 + "\nPART 2 - the map has no stale entries\n" + "=" * 70)
    for owner in sorted(ALLOWED):
        gone = sorted(ALLOWED[owner] - edges[owner])
        check(f"{owner}: every allowed import is still used", not gone,
              f"remove {gone} from ALLOWED[{owner!r}]")

    print("\n" + "=" * 70 + "\nPART 3 - cycles\n" + "=" * 70)
    cycles = strongly_connected(edges)
    unexpected = [c for c in cycles if c not in ACCEPTED_CYCLES]
    check("no cycle between packages other than the accepted one", not unexpected,
          "; ".join(sorted("<->".join(sorted(c)) for c in unexpected)))
    check("the accepted cycle is still there (or the exception can be dropped)",
          all(c in cycles for c in ACCEPTED_CYCLES), "remove it from ACCEPTED_CYCLES")

    print("\n" + "=" * 70 + "\nPART 4 - top of the stack\n" + "=" * 70)
    importers_of = lambda p: sorted(o for o, deps in edges.items() if p in deps)
    check("nothing but api.py imports the HTTP layer", not any("api" in d for d in edges.values()))
    check("nothing but tests imports the evaluation package", importers_of("evaluation") == [],
          str(importers_of("evaluation")))

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE - {len(FAIL)} FAILURE(S): {FAIL}")
        raise SystemExit(1)
    print("DONE - all checks passed.")


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    main()
