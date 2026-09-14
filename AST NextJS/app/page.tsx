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

        {tab === "csv" ? (
          <CsvAnalysisTab file={csvFile} />
        ) : tab === "document" ? (
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
        ) : (
          <MultiTableTab />
        )}
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
