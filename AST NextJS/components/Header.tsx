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
  const now = new Date();
  const dateLabel = now.toISOString().slice(0, 10);
  const timeLabel = now.toISOString().slice(11, 16);

  return (
    <header className="mb-8">
      <p className="font-mono text-xs text-accent tracking-widest mb-2">
        // SECURE ANALYTICS TERMINAL
      </p>
      <h1 className="text-4xl font-bold tracking-tight">
        ANALYST<span className="text-accent">_</span>ASSISTANT
      </h1>
      <p className="text-muted mt-1 mb-4">
        AI-Powered Data Intelligence System — Local Inference Engine
      </p>

      <div className="flex flex-wrap gap-3">
        <Badge label="SYSTEM NOMINAL" />
        <Badge label="LLM READY" />
        <Badge label={`${dateLabel} // ${timeLabel} UTC`} />
      </div>
    </header>
  );
}
