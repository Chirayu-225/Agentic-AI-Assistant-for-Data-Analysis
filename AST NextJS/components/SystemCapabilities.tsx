"use client";

const CAPABILITIES = [
  {
    title: "Ask in plain English",
    description: "Totals, breakdowns, rankings, filters and charts — answered by code run on your file",
  },
  {
    title: "Clean messy exports",
    description: "Currency symbols, spelling variants and total rows fixed only after you opt in",
  },
  {
    title: "Verified insights",
    description: "Auto-Analyze shows a finding only if every number in it matches a computation",
  },
  {
    title: "Show the work",
    description: "Open “How this was computed” on any answer to see the exact code and result",
  },
];

export default function SystemCapabilities() {
  return (
    <div className="border border-border rounded-lg bg-panel p-5">
      <p className="font-mono text-xs text-accent tracking-widest mb-4">// SYSTEM CAPABILITIES</p>
      <div className="flex flex-col divide-y divide-border">
        {CAPABILITIES.map((cap) => (
          <div key={cap.title} className="py-3 flex gap-3">
            <span className="text-accent mt-0.5" aria-hidden>▸</span>
            <div>
              <p className="font-semibold text-text text-sm">{cap.title}</p>
              <p className="text-muted text-xs mt-0.5">{cap.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
