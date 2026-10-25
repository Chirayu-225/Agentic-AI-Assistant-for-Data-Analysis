"use client";

function Badge({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-2 px-3 py-1 rounded-md border border-border
                      bg-panel font-mono text-[11px] text-accent">
      <span className="w-1.5 h-1.5 rounded-full bg-accent inline-block" aria-hidden />
      {label}
    </span>
  );
}

export default function Header() {
  return (
    <header className="mb-8">
      <p className="font-mono text-xs text-accent tracking-widest mb-2">
        // FOR FINANCE &amp; OPERATIONS ANALYSTS
      </p>
      <h1 className="text-4xl font-bold tracking-tight">
        ANALYST<span className="text-accent">_</span>ASSISTANT
      </h1>
      <p className="text-muted mt-1 mb-4">
        Verified answers from your spreadsheets. Every number is computed from your data, and you can see exactly how.
      </p>

      <div className="flex flex-wrap gap-3">
        <Badge label="EVERY NUMBER COMPUTED" />
        <Badge label="CODE SHOWN FOR EVERY ANSWER" />
        <Badge label="PERSONAL COLUMNS WITHHELD FROM THE AI" />
      </div>
    </header>
  );
}
