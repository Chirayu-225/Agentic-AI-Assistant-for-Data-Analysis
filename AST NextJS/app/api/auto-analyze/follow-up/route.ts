/**
 * app/api/auto-analyze/follow-up/route.ts — answers a follow-up question
 * about the insights already shown from Auto-Analyze. See
 * auto_analyze_service.answer_followup's docstring for the two-step
 * decide-then-answer design this exists to trigger.
 */

import { NextRequest, NextResponse } from "next/server";
import { askFollowUp } from "@/lib/pythonClient";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);

  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { fileUrl, filename, sessionId, insights, question, priorQa } = body as Record<string, unknown>;

  if (
    typeof fileUrl !== "string" ||
    typeof filename !== "string" ||
    typeof sessionId !== "string" ||
    typeof question !== "string" ||
    !Array.isArray(insights)
  ) {
    return NextResponse.json(
      { error: "fileUrl, filename, sessionId, question (strings) and insights (array) are required" },
      { status: 400 }
    );
  }

  try {
    const result = await askFollowUp({
      fileUrl,
      filename,
      sessionId,
      insights: insights as { finding: string; action: string }[],
      question,
      priorQa: Array.isArray(priorQa) ? (priorQa as { question: string; answer: string }[]) : [],
    });
    return NextResponse.json(result);
  } catch (err) {
    return NextResponse.json(
      { error: `Follow-up service unavailable: ${(err as Error).message}` },
      { status: 502 }
    );
  }
}
