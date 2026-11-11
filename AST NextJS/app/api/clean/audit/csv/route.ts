/**
 * app/api/clean/audit/csv/route.ts — the whole change list as a CSV download.
 */

import { NextRequest, NextResponse } from "next/server";
import { getAuditCsv, upstreamErrorResponse } from "@/lib/pythonClient";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }
  const b = body as Record<string, unknown>;
  if (typeof b.fileUrl !== "string" || typeof b.filename !== "string" || typeof b.sessionId !== "string") {
    return NextResponse.json({ error: "fileUrl, filename, and sessionId are all required strings" }, { status: 400 });
  }
  try {
    const text = await getAuditCsv({
      fileUrl: b.fileUrl,
      filename: b.filename,
      sessionId: b.sessionId,
      scope: b.scope === "agent" ? "agent" : "clean",
    });
    return new NextResponse(text, {
      status: 200,
      headers: {
        "Content-Type": "text/csv; charset=utf-8",
        "Content-Disposition": 'attachment; filename="audit-trail.csv"',
      },
    });
  } catch (err) {
    return upstreamErrorResponse(err, "Audit service unavailable");
  }
}
