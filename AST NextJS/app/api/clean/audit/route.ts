/**
 * app/api/clean/audit/route.ts — the audit trail: every change the pipeline made, with its row, column, before and
 * after. Read-only; the Python service rebuilds it from the upload (cached) and pages it.
 */

import { NextRequest, NextResponse } from "next/server";
import { getAudit, upstreamErrorResponse } from "@/lib/pythonClient";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }
  const b = body as Record<string, unknown>;
  if (typeof b.fileUrl !== "string" || typeof b.filename !== "string" || typeof b.sessionId !== "string") {
    return NextResponse.json({ error: "fileUrl, filename, and sessionId are all required strings" }, { status: 400 });
  }
  const str = (v: unknown) => (typeof v === "string" && v.length > 0 && v.length <= 200 ? v : undefined);
  const num = (v: unknown, d: number) => (typeof v === "number" && Number.isFinite(v) ? v : d);
  try {
    const result = await getAudit({
      fileUrl: b.fileUrl,
      filename: b.filename,
      sessionId: b.sessionId,
      scope: b.scope === "agent" ? "agent" : "clean",
      offset: num(b.offset, 0),
      limit: num(b.limit, 100),
      rule: str(b.rule),
      column: str(b.column),
      action: str(b.action),
      q: str(b.q),
      refresh: b.refresh === true,
    });
    return NextResponse.json(result);
  } catch (err) {
    return upstreamErrorResponse(err, "Audit service unavailable");
  }
}
