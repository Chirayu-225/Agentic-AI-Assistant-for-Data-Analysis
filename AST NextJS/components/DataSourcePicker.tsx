"use client";

import { useState } from "react";
import FileUpload, { UploadedFile } from "@/components/FileUpload";
import { getOrCreateSessionId } from "@/lib/sessionId";
import type { ConnectorImportResponse } from "@/lib/pythonClient";

type Source = "upload" | "drive" | "mysql";

export interface ImportedDataset extends UploadedFile {
  note?: string;
}

const TABS: { id: Source; label: string }[] = [
  { id: "upload", label: "Upload file" },
  { id: "drive", label: "Google Drive" },
  { id: "mysql", label: "MySQL" },
];

const inputCls =
  "w-full bg-base border border-border rounded-md px-3 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-50";
const labelCls = "font-mono text-[10px] text-muted tracking-widest mb-1 block";

/** Where a dataset for the Join SQL Analysis comes from: a local file, a Google Drive link, or a
 * MySQL table. Whatever the source, the backend stores it and hands back the same
 * {url, filename} an upload gives, so the caller treats all three the same. */
export default function DataSourcePicker({
  onImported,
  onImportedMany,
  datasetNumber,
}: {
  onImported: (file: ImportedDataset) => void;
  /** Called instead of onImported when one import brings back several files (a Drive folder, or several links). */
  onImportedMany?: (files: ImportedDataset[], note?: string) => void;
  datasetNumber: number;
}) {
  const [source, setSource] = useState<Source>("upload");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [driveLink, setDriveLink] = useState("");

  const [host, setHost] = useState("");
  const [port, setPort] = useState("3306");
  const [database, setDatabase] = useState("");
  const [table, setTable] = useState("");
  const [user, setUser] = useState("");
  const [password, setPassword] = useState("");
  const [ssl, setSsl] = useState(false);
  const [limit, setLimit] = useState("");

  async function post(path: string, body: Record<string, unknown>) {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body, sessionId: getOrCreateSessionId() }),
      });
      const data = (await res.json().catch(() => ({}))) as ConnectorImportResponse & { error?: string };
      if (!res.ok || !data.success || !data.url || !data.filename) {
        setError(data.error ?? "Import failed. Please try again.");
        return false;
      }
      if (data.files && data.files.length > 0) {
        const files = data.files.map((f) => ({ url: f.url, filename: f.filename }));
        if (onImportedMany) onImportedMany(files, data.note);
        else files.forEach((f, i) => onImported({ ...f, note: i === 0 ? data.note : undefined }));
        return true;
      }
      onImported({ url: data.url, filename: data.filename, note: data.note });
      return true;
    } catch {
      setError("Could not reach the server. Please try again.");
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function importDrive() {
    if (await post("/api/connect/google-drive", { link: driveLink })) setDriveLink("");
  }

  async function importMySql() {
    const ok = await post("/api/connect/mysql", {
      host: host.trim(), port, database: database.trim(), table: table.trim(), user, password, ssl,
      limit: limit.trim(),
    });
    if (ok) {
      setPassword(""); // never keep the password around once it has been used
      setTable("");
    }
  }

  const mysqlReady = host.trim() && database.trim() && table.trim() && user;

  return (
    <div className="border border-border rounded-md p-3">
      <p className="font-mono text-[11px] text-muted tracking-widest mb-2">// ADD DATASET {datasetNumber}</p>
      <div className="flex gap-1 mb-3" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={source === t.id}
            onClick={() => { setSource(t.id); setError(null); }}
            disabled={busy}
            className={`font-mono text-xs px-3 py-1.5 rounded-md border ${
              source === t.id ? "border-accent text-accent" : "border-border text-muted hover:text-text"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {source === "upload" && (
        <FileUpload
          compact
          onUploaded={(f) => onImported(f)}
          accept=".csv,.xlsx,.xls"
          label="// FROM YOUR COMPUTER"
          helperText="CSV, XLSX, XLS"
        />
      )}

      {source === "drive" && (
        <div className="flex flex-col gap-2">
          <label className={labelCls} htmlFor="drive-link">GOOGLE DRIVE FOLDER, OR FILE / GOOGLE SHEETS LINKS</label>
          <textarea
            id="drive-link"
            value={driveLink}
            onChange={(e) => setDriveLink(e.target.value)}
            onKeyDown={(e) => (e.ctrlKey || e.metaKey) && e.key === "Enter" && driveLink.trim() && !busy && importDrive()}
            disabled={busy}
            rows={3}
            placeholder={"https://drive.google.com/drive/folders/…   (a whole folder)\nhttps://drive.google.com/file/d/…/view   (or one link per line)"}
            className={`${inputCls} resize-y`}
          />
          <p className="font-mono text-[11px] text-muted">
            Paste a <span className="text-text">folder</span> link to import every CSV, Excel file and Google Sheet in it
            (up to 12; subfolders are not opened), or paste several file links, one per line. Share the folder or files as{" "}
            <span className="text-text">&quot;Anyone with the link&quot; (Viewer)</span> first — the app does not
            sign in to your Google account, so private files can&apos;t be read. A Google Sheet imports its
            first tab, or the tab in the link (…#gid=). Ctrl+Enter imports.
          </p>
          <div>
            <button
              onClick={importDrive}
              disabled={busy || !driveLink.trim()}
              className="px-4 py-2 rounded-md bg-accent text-base font-mono text-xs font-semibold disabled:opacity-40"
            >
              {busy ? "Importing…" : "Import from Drive"}
            </button>
          </div>
        </div>
      )}

      {source === "mysql" && (
        <form
          className="flex flex-col gap-3"
          autoComplete="off"
          onSubmit={(e) => { e.preventDefault(); if (mysqlReady && !busy) importMySql(); }}
        >
          <div className="grid grid-cols-3 gap-3">
            <div className="col-span-2">
              <label className={labelCls} htmlFor="my-host">HOST</label>
              <input id="my-host" value={host} onChange={(e) => setHost(e.target.value)} disabled={busy}
                placeholder="localhost or db.example.com" className={inputCls} />
            </div>
            <div>
              <label className={labelCls} htmlFor="my-port">PORT</label>
              <input id="my-port" value={port} onChange={(e) => setPort(e.target.value)} disabled={busy}
                inputMode="numeric" className={inputCls} />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className={labelCls} htmlFor="my-db">DATABASE</label>
              <input id="my-db" value={database} onChange={(e) => setDatabase(e.target.value)} disabled={busy}
                placeholder="shop" className={inputCls} />
            </div>
            <div>
              <label className={labelCls} htmlFor="my-table">TABLE NAME</label>
              <input id="my-table" value={table} onChange={(e) => setTable(e.target.value)} disabled={busy}
                placeholder="orders" className={inputCls} />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className={labelCls} htmlFor="my-user">USERNAME</label>
              <input id="my-user" value={user} onChange={(e) => setUser(e.target.value)} disabled={busy}
                autoComplete="off" className={inputCls} />
            </div>
            <div>
              <label className={labelCls} htmlFor="my-pass">PASSWORD</label>
              <input id="my-pass" type="password" value={password} onChange={(e) => setPassword(e.target.value)}
                disabled={busy} autoComplete="new-password" className={inputCls} />
            </div>
          </div>
          <div className="flex items-center gap-4 flex-wrap">
            <label className="font-mono text-xs text-text flex items-center gap-2">
              <input type="checkbox" checked={ssl} onChange={(e) => setSsl(e.target.checked)} disabled={busy} />
              Use SSL
            </label>
            <label className="font-mono text-xs text-muted flex items-center gap-2">
              Max rows
              <input value={limit} onChange={(e) => setLimit(e.target.value)} disabled={busy} inputMode="numeric"
                placeholder="200000" className={`${inputCls} w-28`} />
            </label>
          </div>
          <p className="font-mono text-[11px] text-muted">
            Reads one whole table, read-only. Your password is used for this import only and is never saved or
            logged. Use an account that can only read this database. To join several tables, import each one.
          </p>
          <div>
            <button
              type="submit"
              disabled={busy || !mysqlReady}
              className="px-4 py-2 rounded-md bg-accent text-base font-mono text-xs font-semibold disabled:opacity-40"
            >
              {busy ? "Importing…" : "Import table"}
            </button>
          </div>
        </form>
      )}

      {error && <p className="font-mono text-xs text-red-400 mt-3">⚠ {error}</p>}
    </div>
  );
}
