"use client";

const CAPABILITIES = [
  {
    title: "Statistical Analysis",
    description: "Correlations, distributions, descriptive stats, outlier detection",
  },
  {
    title: "Data Filtering & Aggregation",
    description: "Group by, filter, sort, pivot — all via natural language",
  },
  {
    title: "Pattern Recognition",
    description: "Trends, seasonality, anomaly identification across dimensions",
  },
  {
    title: "Code Generation & Execution",
    description: "Auto-generates pandas code, runs it in a sandboxed process, surfaces results",
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
