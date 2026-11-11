import sys, time, resource; sys.path.insert(0, ".")
import audit_service as svc, session_cache
path, name, scope = sys.argv[1], sys.argv[2], sys.argv[3]
data = open(path, "rb").read()
t = time.time()
L, s = svc.build_audit(data, name, "s", "u", scope)
print(f"{name} [{scope}]: {time.time()-t:.1f}s  peakMB {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024}")
print("totals:", s["totals"])
print("stages:", s["stages"], "| entries stored:", s["entryCount"], "| truncated:", s["truncated"], "| conversions:", [(c["column"], c["to"], c["count"]) for c in s["conversions"]])
print("unexplained:", s["unexplained"])
print("duplicates:", {k: v for k, v in s["duplicates"].items() if k != "sample"}, [g["rows"][:6] for g in s["duplicates"]["sample"][:2]])
for r in s["byRule"][:14]:
    print("  ", r["count"], "|", r["stage"], "|", r["rule"], "|", r["column"], "|", r["action"], "|", (r["evidence"] or "")[:70])
for e in L.page(action="removed", limit=3)["entries"]: print("  removed:", e)
