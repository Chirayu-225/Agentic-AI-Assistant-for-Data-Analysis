"use client";

import { useRef, useState } from "react";
import { upload } from "@vercel/blob/client";
import { getOrCreateSessionId } from "@/lib/sessionId";

export interface UploadedFile {
  url: string;
  filename: string;
}

export default function FileUpload({
  onUploaded,
  accept = ".csv",
  label = "// DATA INGEST — CSV",
  helperText = "Limit 200MB per file • CSV",
  compact = false,
}: {
  onUploaded: (file: UploadedFile) => void;
  /** File picker filter, e.g. ".csv" or ".pdf,.docx,.txt,.md" */
  accept?: string;
  /** Small header line above the picker — kept generic so this component
   * works for both the CSV and Document Intelligence tabs. */
  label?: string;
  helperText?: string;
  /** Sidebar-style compact rendering — small button instead of the full
   * bordered panel used in the main content area. */
  compact?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadedName, setUploadedName] = useState<string | null>(null);

  async function handleFileChange() {
    const file = inputRef.current?.files?.[0];
    if (!file) return;

    setError(null);
    setProgress(0);

    try {
      const sessionId = getOrCreateSessionId();
      const blob = await upload(file.name, file, {
        access: "public", // known trade-off — see README's "Blob access" note
        handleUploadUrl: "/api/upload",
        clientPayload: sessionId, // carried through to the server's onUploadCompleted
        onUploadProgress: (evt: { percentage: number }) => setProgress(evt.percentage),
      });

      setUploadedName(file.name);
      onUploaded({ url: blob.url, filename: file.name });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setProgress(null);
    }
  }

  if (compact) {
    return (
      <div>
        <p className="font-mono text-xs text-muted mb-2 tracking-wide">{label}</p>
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          onChange={handleFileChange}
          className="hidden"
          id={`file-input-${label}`}
        />
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="flex items-center gap-2 px-3 py-2 rounded-md border border-border
                     bg-base text-accent font-mono text-xs hover:bg-border transition-colors"
        >
          <span aria-hidden>↑</span> Upload
        </button>
        {progress !== null && (
          <p className="text-accent font-mono text-[11px] mt-2">Uploading — {progress}%</p>
        )}
        {uploadedName && progress === null && !error && (
          <p className="text-accent font-mono text-[11px] mt-2 truncate">✓ {uploadedName}</p>
        )}
        {error && <p className="text-red-400 font-mono text-[11px] mt-2">{error}</p>}
        <p className="text-muted text-[11px] mt-2">{helperText}</p>
      </div>
    );
  }

  return (
    <div className="border border-border rounded-lg p-6 bg-panel">
      <p className="font-mono text-sm text-muted mb-3">{label}</p>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={handleFileChange}
        className="block w-full text-sm text-text file:mr-4 file:py-2 file:px-4
                   file:rounded-md file:border file:border-border
                   file:bg-base file:text-accent file:font-mono
                   hover:file:bg-border"
      />
      {progress !== null && (
        <p className="text-accent font-mono text-xs mt-2">Uploading — {progress}%</p>
      )}
      {error && <p className="text-red-400 font-mono text-xs mt-2">{error}</p>}
      <p className="text-muted text-xs mt-2">{helperText}</p>
    </div>
  );
}
