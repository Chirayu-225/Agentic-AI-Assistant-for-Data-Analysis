/**
 * app/api/data-checks/route.ts — deterministic format / encoding checks on the
 * uploaded table. Mirrors app/api/clean/route.ts. The UI treats any failure
 * as "no note", so this never blocks the analysis flow.
 */

import { NextRequest, NextResponse } from "next/server";
import { getDataChecks, upstreamErrorResponse } from "@/lib/pythonClient";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);

  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { fileUrl, filename } = body as Record<string, unknown>;

  if (typeof fileUrl !== "string" || typeof filename !== "string") {
    return NextResponse.json(
      { error: "fileUrl and filename are both required strings" },
      { status: 400 }
    );
  }

  try {
    const result = await getDataChecks({ fileUrl, filename });
    return NextResponse.json(result);
  } catch (err) {
    return upstreamErrorResponse(err, "Data checks unavailable");
  }
}
