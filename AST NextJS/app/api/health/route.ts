/**
 * app/api/health/route.ts — live status for the sidebar: is the Python
 * backend reachable, and which models is it configured with. Replaces
 * hard-coded "ONLINE / AES-256 / 99.97% uptime" values that nothing measured.
 */

import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  const base = process.env.PYTHON_SERVICE_URL?.replace(/\/+$/, "");
  if (!base) return NextResponse.json({ online: false, reason: "PYTHON_SERVICE_URL not set" });
  try {
    const res = await fetch(`${base}/health`, { cache: "no-store", signal: AbortSignal.timeout(4000) });
    if (!res.ok) return NextResponse.json({ online: false, reason: `backend returned ${res.status}` });
    const data = (await res.json()) as {
      models?: { code?: string; chat?: string };
      sandboxMemoryCap?: boolean;
    };
    return NextResponse.json({ online: true, models: data.models ?? null, sandboxMemoryCap: data.sandboxMemoryCap ?? null });
  } catch {
    return NextResponse.json({ online: false, reason: "backend unreachable" });
  }
}
