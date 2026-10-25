"use client";

import { Suspense, useState, type FormEvent } from "react";
import { useRouter, useSearchParams } from "next/navigation";

function AccessForm() {
  const router = useRouter();
  const params = useSearchParams();
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);

  // Only follow same-site relative paths after sign-in (no open redirect).
  const raw = params.get("next") ?? "/";
  const next = raw.startsWith("/") && !raw.startsWith("//") ? raw : "/";

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!code.trim() || checking) return;
    setChecking(true);
    setError(null);
    try {
      const res = await fetch("/api/access", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      const data = (await res.json().catch(() => ({}))) as { success?: boolean; error?: string };
      if (res.ok && data.success) {
        router.replace(next);
        router.refresh();
      } else {
        setError(data.error ?? "Could not check the code.");
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setChecking(false);
    }
  }

  return (
    <form onSubmit={submit} className="w-full max-w-sm border border-border rounded-lg bg-panel p-6">
      <p className="font-mono text-xs text-accent tracking-widest mb-2">// ACCESS REQUIRED</p>
      <h1 className="text-2xl font-bold tracking-tight mb-1">
        ANALYST<span className="text-accent">_</span>ASSISTANT
      </h1>
      <p className="text-muted text-sm mb-5">Enter the access code you were given to use this demo.</p>
      <input
        type="password"
        autoFocus
        autoComplete="current-password"
        value={code}
        onChange={(e) => setCode(e.target.value)}
        placeholder="Access code"
        aria-label="Access code"
        className="w-full bg-base border border-border rounded-md px-3 py-2 font-mono text-sm text-text placeholder:text-muted focus:outline-none focus:border-accent mb-3"
      />
      <button
        type="submit"
        disabled={checking || !code.trim()}
        className="w-full px-4 py-2 rounded-md bg-accent text-base font-mono text-sm font-semibold disabled:opacity-40"
      >
        {checking ? "Checking..." : "Enter"}
      </button>
      {error && <p className="text-red-400 text-xs mt-3">{error}</p>}
    </form>
  );
}

export default function AccessPage() {
  return (
    <main className="min-h-screen flex items-center justify-center px-4">
      <Suspense fallback={null}>
        <AccessForm />
      </Suspense>
    </main>
  );
}
