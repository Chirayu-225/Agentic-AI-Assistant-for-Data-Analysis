"use client";

import { useState, type ReactNode } from "react";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import CsvAnalysisTab from "@/components/CsvAnalysisTab";
import DocumentIntelligenceTab from "@/components/DocumentIntelligenceTab";
import MultiTableTab from "@/components/MultiTableTab";
import type { UploadedFile } from "@/components/FileUpload";

type Tab = "csv" | "document" | "multi";

export default function Home() {
  const [tab, setTab] = useState<Tab>("csv");
  const [csvFile, setCsvFile] = useState<UploadedFile | null>(null);
  const [docFile, setDocFile] = useState<UploadedFile | null>(null);

  return (
    <div className="min-h-screen flex">
      <Sidebar
        onCsvUploaded={(f: UploadedFile) => {
          setCsvFile(f);
          setTab("csv");
        }}
        onDocUploaded={(f: UploadedFile) => {
          setDocFile(f);
          setTab("document");
        }}
      />

      <main className="flex-1 min-w-0 px-8 py-10">
        <Header />

        <div className="flex gap-6 border-b border-border mb-6">
          <TabButton active={tab === "csv"} onClick={() => setTab("csv")}>
            ⚡ CSV Analysis
          </TabButton>
          <TabButton active={tab === "document"} onClick={() => setTab("document")}>
            📄 Document Intelligence
          </TabButton>
          <TabButton active={tab === "multi"} onClick={() => setTab("multi")}>
            🔗 Multi-Dataset Query
          </TabButton>
        </div>

        {/*
          All three tabs stay MOUNTED at all times — only CSS visibility
          switches between them (`hidden` vs not), rather than the old
          ternary that rendered just one and let React unmount the
          others.

          WHY THIS CHANGED (live bug): CsvAnalysisTab and
          DocumentIntelligenceTab each run a real network call inside a
          useEffect keyed on their `file`/`file` prop the moment they
          mount (/api/profile, /api/rag/index respectively) — see those
          components' own effects. Unmounting on tab-switch and
          remounting on switch-back doesn't preserve React state across
          that gap; it throws it away, so the component's effect fires
          again from scratch as if it were seeing the file for the first
          time. Concretely: pick a table on the Document Intelligence
          tab (→ CSV Analysis, correct), then click back to Document
          Intelligence — the old code unmounted DocumentIntelligenceTab
          when you left and remounted a BRAND NEW instance when you
          returned, which re-ran /api/rag/index (re-parsing, re-chunking,
          re-embedding the same PDF) instead of just showing the table
          list you'd already gotten back — from the user's side, "you
          can't go back to choosing a different table," since it looked
          like the app lost track of what it already found until that
          full re-index finished. Keeping all three mounted and just
          hiding the inactive ones with CSS fixes this for every tab
          pair, not only Document Intelligence → CSV, and is the
          standard fix for "switching tabs shouldn't cost a re-fetch."
        */}
        <div className={tab === "csv" ? "" : "hidden"}>
          <CsvAnalysisTab file={csvFile} />
        </div>
        <div className={tab === "document" ? "" : "hidden"}>
          <DocumentIntelligenceTab
            file={docFile}
            onAnalyzeTable={(table) => {
              // A table extracted from a PDF (table_extraction.py) is an
              // ordinary CSV Blob URL by the time it gets here — handing
              // it to the SAME CsvAnalysisTab a regular upload uses, no
              // different code path, so it gets full profiling/cleaning/
              // auto-analyze/follow-up for free.
              setCsvFile(table);
              setTab("csv");
            }}
          />
        </div>
        <div className={tab === "multi" ? "" : "hidden"}>
          <MultiTableTab />
        </div>
      </main>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`pb-3 px-1 font-mono text-sm border-b-2 transition-colors ${
        active
          ? "border-accent text-accent"
          : "border-transparent text-muted hover:text-text"
      }`}
    >
      {children}
    </button>
  );
}
