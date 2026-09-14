/**
 * app/api/auto-analyze/route.ts — triggers the proactive dataset
 * intelligence pipeline: up to 5 ranked business insights (each with a
 * concrete action) plus up to 3 charts chosen specifically to make those
 * insights visible at a glance. Explicitly separate from any automatic
 * post-upload step — like /api/clean/agent, this costs several real LLM
 * calls, so the user should knowingly opt in via the button rather than
 * have it fire silently on every upload.
 */

import { NextRequest, NextResponse } from "next/server";
import { runAutoAnalyze } from "@/lib/pythonClient";

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
    const result = await runAutoAnalyze({ fileUrl, filename, sessionId });
    return NextResponse.json(result);
  } catch (err) {
    return NextResponse.json(
      { error: `Auto-analyze service unavailable: ${(err as Error).message}` },
      { status: 502 }
    );
  }
}
