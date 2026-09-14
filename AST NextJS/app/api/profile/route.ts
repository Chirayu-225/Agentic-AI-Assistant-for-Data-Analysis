/**
 * app/api/profile/route.ts — raw (pre-cleaning) dataset stats, shown
 * immediately after CSV upload. Mirrors app/api/analyze/route.ts.
 */

import { NextRequest, NextResponse } from "next/server";
import { getProfile } from "@/lib/pythonClient";

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
    const result = await getProfile({ fileUrl, filename });
    return NextResponse.json(result);
  } catch (err) {
    return NextResponse.json(
      { error: `Profiling service unavailable: ${(err as Error).message}` },
      { status: 502 }
    );
  }
}
