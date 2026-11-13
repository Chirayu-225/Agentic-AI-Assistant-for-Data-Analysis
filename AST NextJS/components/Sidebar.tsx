"use client";

import { useEffect, useState } from "react";
import { useClock } from "@/lib/useClock";
import FileUpload, { UploadedFile } from "@/components/FileUpload";

interface StatRowProps {
  label: string;
  value: string;
  valueClassName?: string;
}

function StatRow({ label, value, valueClassName = "text-accent" }: StatRowProps) {
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="font-mono text-[11px] text-muted tracking-wide">{label}</span>
      <span className={`font-mono text-[11px] font-semibold ${valueClassName}`}>{value}</span>
    </div>
  );
}

interface Health {
  online: boolean;
  models?: { code?: string; chat?: string } | null;
  sandboxMemoryCap?: boolean | null;
}

// "openai/gpt-oss-120b" -> "GPT-OSS 120B"
function prettyModel(id?: string): string {
  if (!id) return "—";
  const name = id.split("/").pop() ?? id;
  return name.replace(/-(\d+b)$/i, " $1").toUpperCase();
}

/** Polls /api/health so the sidebar shows real status, not decoration. */
function useBackendHealth(): Health | null {
  const [health, setHealth] = useState<Health | null>(null);
  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const res = await fetch("/api/health", { cache: "no-store" });
        const data = (await res.json()) as Health;
        if (!cancelled) setHealth(data);
      } catch {
        if (!cancelled) setHealth({ online: false });
      }
    }
    check();
    const id = setInterval(check, 30_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);
  return health;
}

export default function Sidebar({
  onCsvUploaded,
  onDocUploaded,
}: {
  onCsvUploaded: (file: UploadedFile) => void;
  onDocUploaded: (file: UploadedFile) => void;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const time = useClock();
  const health = useBackendHealth();

  if (collapsed) {
    return (
      <button
        onClick={() => setCollapsed(false)}
        className="w-10 shrink-0 border-r border-border flex items-start justify-center pt-6
                   text-muted hover:text-accent transition-colors"
        aria-label="Expand sidebar"
      >
        »
      </button>
    );
  }

  return (
    <aside className="w-72 shrink-0 border-r border-border px-5 py-6 flex flex-col gap-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="font-mono text-xs font-bold text-accent tracking-wide">
            ANALYST ASSISTANT
          </p>
          <p className="font-mono text-[10px] text-muted mt-0.5">v1.0</p>
        </div>
        <button
          onClick={() => setCollapsed(true)}
          className="text-muted hover:text-accent transition-colors font-mono text-xs"
          aria-label="Collapse sidebar"
        >
          «
        </button>
      </div>

      <div className="border border-border rounded-md px-3 py-2 bg-panel">
        <p className="font-mono text-2xl font-bold text-accent tracking-wider">{time}</p>
      </div>

      <div className="border-t border-border pt-3">
        {/* Every row here is either measured live (/api/health) or a true
            statement about the code — no decorative security/uptime claims. */}
        <StatRow
          label="BACKEND"
          value={health === null ? "CHECKING…" : health.online ? "ONLINE" : "OFFLINE"}
          valueClassName={health === null ? "text-muted" : health.online ? "text-accent" : "text-red-400"}
        />
        <StatRow label="CODE_MODEL" value={health?.online ? prettyModel(health.models?.code) : "—"} />
        <StatRow label="CHAT_MODEL" value={health?.online ? prettyModel(health.models?.chat) : "—"} />
        <StatRow label="INFERENCE" value="GROQ API" />
        <StatRow label="EMBEDDINGS" value="LOCAL · CPU" />
      </div>

      <div className="border-t border-border pt-4 flex flex-col gap-5">
        <FileUpload
          compact
          onUploaded={onCsvUploaded}
          accept=".csv,.xlsx,.xls"
          label="// DATA INGEST — CSV / EXCEL"
          helperText="200MB per file • CSV, XLSX, XLS"
        />
        <FileUpload
          compact
          onUploaded={onDocUploaded}
          accept=".pdf,.docx,.txt,.md"
          label="// DATA INGEST — DOCUMENT"
          helperText="200MB per file • PDF, DOCX, TXT..."
        />
      </div>
    </aside>
  );
}
