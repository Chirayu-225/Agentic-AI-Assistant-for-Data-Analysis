/**
 * app/api/clean/duplicates/route.ts — the user's answer to "keep or remove the identical rows": drops the repeats
 * (the first copy of each stays) from the table the questions use. Explicit click only.
 */

import { NextRequest, NextResponse } from "next/server";
import { removeDuplicates, upstreamErrorResponse } from "@/lib/pythonClient";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }
  const { fileUrl, filename, sessionId } = body as Record<string, unknown>;
  if (typeof fileUrl !== "string" || typeof filename !== "string" || typeof sessionId !== "string") {
    return NextResponse.json({ error: "fileUrl, filename, and sessionId are all required strings" }, { status: 400 });
  }
  try {
    return NextResponse.json(await removeDuplicates({ fileUrl, filename, sessionId }));
  } catch (err) {
    return upstreamErrorResponse(err, "Duplicate removal unavailable");
  }
}
