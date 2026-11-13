/**
 * app/api/clean/scale/route.ts — the user's "yes" to the x1000 question: divides the values that are exactly 1000 times
 * too large by 1000 in the working table. Explicit click only; no model is involved.
 */

import { NextRequest, NextResponse } from "next/server";
import { scaleFix, upstreamErrorResponse } from "@/lib/pythonClient";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }
  const { fileUrl, filename, sessionId, columns } = body as Record<string, unknown>;
  if (typeof fileUrl !== "string" || typeof filename !== "string" || typeof sessionId !== "string") {
    return NextResponse.json({ error: "fileUrl, filename, and sessionId are all required strings" }, { status: 400 });
  }
  const cols = Array.isArray(columns) ? columns.filter((c): c is string => typeof c === "string") : [];
  try {
    return NextResponse.json(await scaleFix({ fileUrl, filename, sessionId, columns: cols }));
  } catch (err) {
    return upstreamErrorResponse(err, "Scale fix unavailable");
  }
}
