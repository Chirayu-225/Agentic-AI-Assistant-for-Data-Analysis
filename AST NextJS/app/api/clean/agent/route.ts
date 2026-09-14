/**
 * app/api/clean/agent/route.ts — triggers the LLM-driven cleaning agent
 * for nulls rule-based cleaning couldn't resolve. Explicitly separate
 * from /api/clean (which is free, instant, deterministic) since this
 * costs a real LLM call and isn't perfectly deterministic — the user
 * should knowingly opt in, not have it run automatically.
 */

import { NextRequest, NextResponse } from "next/server";
import { runCleaningAgent } from "@/lib/pythonClient";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);

  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { fileUrl, filename, sessionId } = body as Record<string, unknown>;

  if (
    typeof fileUrl !== "string" ||
    typeof filename !== "string" ||
    typeof sessionId !== "string"
  ) {
    return NextResponse.json(
      { error: "fileUrl, filename, and sessionId are all required strings" },
      { status: 400 }
    );
  }

  try {
    const result = await runCleaningAgent({ fileUrl, filename, sessionId });
    return NextResponse.json(result);
  } catch (err) {
    return NextResponse.json(
      { error: `Cleaning agent service unavailable: ${(err as Error).message}` },
      { status: 502 }
    );
  }
}
