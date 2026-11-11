"use client";

/**
 * AuditDashboard — every change the pipeline made, with the row it was made on.
 * Loads on demand (the first build can take 10–30 s on a large file), then pages the change list from the server.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { AuditEntry, AuditResponse, AuditSummary } from "@/lib/pythonClient";

const PAGE = 50;
const ACTIONS = ["changed", "blanked", "filled", "removed"] as const;

const ACTION_STYLE: Record<string, string> = {
  changed: "text-accent",
  blanked: "text-yellow-400",
  filled: "text-green-400",
  removed: "text-red-400",
};

function Tile({ label, value, bad }: { label: string; value: number | string; bad?: boolean }) {
  return (
    <div className={`border rounded-md p-3 ${bad ? "border-red-700 bg-red-950/30" : "border-border bg-base"}`}>
      <p className={`font-mono text-xl font-bold ${bad ? "text-red-400" : "text-text"}`}>
        {typeof value === "number" ? value.toLocaleString() : value}
      </p>
      <p className="font-mono text-[10px] text-muted tracking-widest mt-1">{label}</p>
    </div>
  );
}

export default function AuditDashboard({
  fileUrl,
  filename,
  sessionId,
  scope,
  onDuplicatesRemoved,
}: {
  fileUrl: string;
  filename: string;
  sessionId: string;
  scope: "clean" | "agent";
  onDuplicatesRemoved?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<AuditSummary | null>(null);
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [action, setAction] = useState("");
  const [rule, setRule] = useState("");
  const [column, setColumn] = useState("");
  const [q, setQ] = useState("");
  const [dupBusy, setDupBusy] = useState(false);
  const [dupMsg, setDupMsg] = useState<string | null>(null);
  const reqId = useRef(0);

  const load = useCallback(
    async (opts: { offset: number; refresh?: boolean }) => {
      const id = ++reqId.current;
      setLoading(true);
      setError(null);
      try {
        const res = await fetch("/api/clean/audit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            fileUrl,
            filename,
            sessionId,
            scope,
            offset: opts.offset,
            limit: PAGE,
            action: action || undefined,
            rule: rule || undefined,
            column: column || undefined,
            q: q || undefined,
            refresh: opts.refresh === true,
          }),
        });
        const data = (await res.json()) as AuditResponse;
        if (id !== reqId.current) return; // a newer request superseded this one
        if (!res.ok || !data.success || !data.summary) {
          setError(data.error ?? "Could not build the audit trail.");
          return;
        }
        setSummary(data.summary);
        setEntries(data.entries ?? []);
        setTotal(data.total ?? 0);
        setOffset(data.offset ?? opts.offset);
      } catch {
        if (id === reqId.current) setError("Could not reach the audit service.");
      } finally {
        if (id === reqId.current) setLoading(false);
      }
    },
    [fileUrl, filename, sessionId, scope, action, rule, column, q],
  );

  // Open → first load. Filters → reload from page 0 (search is debounced).
  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => void load({ offset: 0 }), q ? 300 : 0);
    return () => clearTimeout(t);
  }, [open, load, q]);

  // The cleaned file changed (new scope, e.g. the agent ran) — drop what we show.
  useEffect(() => {
    setSummary(null);
    setEntries([]);
    setDupMsg(null);
  }, [fileUrl, scope]);

  async function downloadCsv() {
    try {
      const res = await fetch("/api/clean/audit/csv", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fileUrl, filename, sessionId, scope }),
      });
      if (!res.ok) throw new Error();
      const url = URL.createObjectURL(await res.blob());
      const a = document.createElement("a");
      a.href = url;
      a.download = "audit-trail.csv";
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setError("Could not download the audit trail.");
    }
  }

  async function removeDupes() {
    setDupBusy(true);
    setDupMsg(null);
    try {
      const res = await fetch("/api/clean/duplicates", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fileUrl, filename, sessionId }),
      });
      const data = await res.json();
      if (!res.ok || !data.success) {
        setDupMsg(data.error ?? "Could not remove the duplicates.");
      } else {
        setDupMsg(data.message ?? `Removed ${data.removed} duplicate rows.`);
        onDuplicatesRemoved?.();
        void load({ offset: 0, refresh: true });
      }
    } catch {
      setDupMsg("Could not remove the duplicates.");
    } finally {
      setDupBusy(false);
    }
  }

  if (!open) {
    return (
      <div className="border border-border rounded-lg bg-panel p-4 mb-6">
        <p className="font-mono text-xs text-muted mb-3">
          See exactly which rows and cells were changed, blanked, filled or removed, and why.
        </p>
        <button
          onClick={() => setOpen(true)}
          className="px-4 py-2 rounded-md border border-accent/60 text-accent font-mono text-sm"
        >
          Open audit trail
        </button>
      </div>
    );
  }

  const t = summary?.totals;
  const rules = summary ? Array.from(new Set(summary.byRule.map((r) => r.rule))) : [];
  const columns = summary ? Array.from(new Set(summary.byRule.map((r) => r.column).filter((c): c is string => !!c))) : [];
  const unexplained = summary?.unexplained.count ?? 0;
  const dups = summary?.duplicates;

  return (
    <div className="border border-accent/40 rounded-lg bg-panel p-5 mb-6">
      <div className="flex items-center justify-between mb-3">
        <p className="font-mono text-xs text-accent tracking-widest">// AUDIT TRAIL</p>
        <div className="flex gap-2">
          <button onClick={downloadCsv} disabled={!summary} className="px-3 py-1 rounded border border-border font-mono text-xs disabled:opacity-40">
            Download CSV
          </button>
          <button onClick={() => setOpen(false)} className="px-3 py-1 rounded border border-border font-mono text-xs">
            Close
          </button>
        </div>
      </div>

      {error && <p className="font-mono text-xs text-red-400 mb-3">{error}</p>}
      {!summary && loading && (
        <p className="font-mono text-sm text-accent animate-pulse">▸ Building the audit trail… (large files can take up to 30 s)</p>
      )}

      {summary && t && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-2 mb-2">
            <Tile label="CELLS CHANGED" value={t.cellsChanged} />
            <Tile label="CELLS BLANKED" value={t.cellsBlanked} />
            <Tile label="CELLS FILLED" value={t.cellsFilled} />
            <Tile label="ROWS REMOVED" value={t.rowsRemoved} />
            <Tile label="ROWS TOUCHED" value={t.rowsTouched} />
            <Tile label="UNEXPLAINED" value={unexplained} bad={unexplained > 0} />
          </div>
          <p className="font-mono text-[11px] text-muted mb-4">
            {t.rowsRaw.toLocaleString()} rows in your file → {t.rowsFinal.toLocaleString()} after cleaning. Row numbers are as
            your spreadsheet shows them (header = row 1).
            {unexplained === 0
              ? " Every difference between your file and the cleaned table is accounted for below."
              : " Some differences are not explained by any recorded fix — see the first examples under the list."}
          </p>
          {summary.notes.map((n, i) => (
            <p key={i} className="font-mono text-[11px] text-yellow-400 mb-1">
              {n}
            </p>
          ))}
          {summary.truncated && (
            <p className="font-mono text-[11px] text-yellow-400 mb-3">
              Only the first {summary.entryCount.toLocaleString()} changes are listed; the counts above are exact.
            </p>
          )}

          {dups && dups.groups > 0 && (
            <div className="border border-border rounded-md p-3 mb-4 bg-base">
              <p className="font-mono text-xs text-text mb-1">
                {dups.extra.toLocaleString()} exact copy row{dups.extra === 1 ? "" : "s"} in {dups.groups.toLocaleString()} group
                {dups.groups === 1 ? "" : "s"} — kept for now.
              </p>
              {dups.sample.slice(0, 5).map((g, i) => (
                <p key={i} className="font-mono text-[11px] text-muted">
                  rows {g.rows.slice(0, 8).join(", ")}
                  {g.rows.length > 8 ? ", …" : ""} are identical
                  {Object.keys(g.values).length > 0 &&
                    ` (${Object.entries(g.values)
                      .slice(0, 3)
                      .map(([k, v]) => `${k}: ${v}`)
                      .join(", ")})`}
                </p>
              ))}
              <div className="mt-2 flex items-center gap-3">
                <button
                  onClick={removeDupes}
                  disabled={dupBusy}
                  className="px-3 py-1 rounded bg-accent text-base font-mono text-xs font-semibold disabled:opacity-50"
                >
                  {dupBusy ? "Removing…" : `Remove ${dups.extra.toLocaleString()} duplicate rows`}
                </button>
                {dupMsg && <span className="font-mono text-[11px] text-text">{dupMsg}</span>}
              </div>
            </div>
          )}

          <div className="overflow-x-auto mb-4">
            <table className="w-full font-mono text-[11px]">
              <thead>
                <tr className="text-muted text-left">
                  <th className="pr-3 py-1">FIX</th>
                  <th className="pr-3">ACTION</th>
                  <th className="pr-3">COLUMN</th>
                  <th className="pr-3 text-right">CELLS / ROWS</th>
                  <th>EVIDENCE</th>
                </tr>
              </thead>
              <tbody>
                {summary.byRule.map((r, i) => (
                  <tr
                    key={i}
                    className="border-t border-border cursor-pointer hover:bg-base"
                    onClick={() => {
                      setRule(r.rule);
                      setColumn(r.column ?? "");
                      setAction(r.action);
                    }}
                  >
                    <td className="pr-3 py-1 text-text">{r.rule}</td>
                    <td className={`pr-3 ${ACTION_STYLE[r.action] ?? ""}`}>{r.action}</td>
                    <td className="pr-3 text-muted">{r.column ?? "—"}</td>
                    <td className="pr-3 text-right text-text">{r.count.toLocaleString()}</td>
                    <td className="text-muted">{r.evidence}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-[10px] text-muted mt-1">Click a line to list just those rows.</p>
          </div>

          <div className="flex flex-wrap gap-2 mb-3">
            <select value={action} onChange={(e) => setAction(e.target.value)} className="bg-base border border-border rounded px-2 py-1 font-mono text-xs">
              <option value="">all actions</option>
              {ACTIONS.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
            <select value={rule} onChange={(e) => setRule(e.target.value)} className="bg-base border border-border rounded px-2 py-1 font-mono text-xs">
              <option value="">all fixes</option>
              {rules.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
            <select value={column} onChange={(e) => setColumn(e.target.value)} className="bg-base border border-border rounded px-2 py-1 font-mono text-xs">
              <option value="">all columns</option>
              {columns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="search a value or row id"
              className="bg-base border border-border rounded px-2 py-1 font-mono text-xs flex-1 min-w-[160px]"
            />
            {(action || rule || column || q) && (
              <button
                onClick={() => {
                  setAction("");
                  setRule("");
                  setColumn("");
                  setQ("");
                }}
                className="px-2 py-1 font-mono text-xs text-muted"
              >
                clear
              </button>
            )}
          </div>

          <div className={`overflow-x-auto ${loading ? "opacity-50" : ""}`}>
            <table className="w-full font-mono text-[11px]">
              <thead>
                <tr className="text-muted text-left">
                  <th className="pr-3 py-1">ROW</th>
                  <th className="pr-3">KEY</th>
                  <th className="pr-3">COLUMN</th>
                  <th className="pr-3">BEFORE</th>
                  <th className="pr-3">AFTER</th>
                  <th className="pr-3">ACTION</th>
                  <th>FIX</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((e, i) => (
                  <tr key={i} className="border-t border-border">
                    <td className="pr-3 py-1 text-text">{e.row}</td>
                    <td className="pr-3 text-muted">{e.key}</td>
                    <td className="pr-3 text-muted">{e.column ?? "(whole row)"}</td>
                    <td className="pr-3 text-muted line-through">{e.before || "∅"}</td>
                    <td className="pr-3 text-text">{e.after || "∅"}</td>
                    <td className={`pr-3 ${ACTION_STYLE[e.action] ?? ""}`}>{e.action}</td>
                    <td className="text-muted">{e.rule}</td>
                  </tr>
                ))}
                {entries.length === 0 && !loading && (
                  <tr>
                    <td colSpan={7} className="py-3 text-muted">
                      No changes match.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between mt-3 font-mono text-[11px] text-muted">
            <span>
              {total === 0 ? 0 : offset + 1}–{Math.min(offset + PAGE, total)} of {total.toLocaleString()}
            </span>
            <div className="flex gap-2">
              <button
                disabled={offset === 0 || loading}
                onClick={() => void load({ offset: Math.max(0, offset - PAGE) })}
                className="px-3 py-1 rounded border border-border disabled:opacity-40"
              >
                Prev
              </button>
              <button
                disabled={offset + PAGE >= total || loading}
                onClick={() => void load({ offset: offset + PAGE })}
                className="px-3 py-1 rounded border border-border disabled:opacity-40"
              >
                Next
              </button>
            </div>
          </div>

          {unexplained > 0 && summary.unexplained.examples.length > 0 && (
            <div className="mt-4 border border-red-700 rounded-md p-3 bg-red-950/30">
              <p className="font-mono text-xs text-red-400 mb-1">
                {unexplained.toLocaleString()} difference{unexplained === 1 ? "" : "s"} with no recorded reason:
              </p>
              {summary.unexplained.examples.slice(0, 5).map((u, i) => (
                <p key={i} className="font-mono text-[11px] text-red-300">
                  row {u.row}, {u.column ?? "?"}: {u.before || "∅"} → {u.after || "∅"}
                </p>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
