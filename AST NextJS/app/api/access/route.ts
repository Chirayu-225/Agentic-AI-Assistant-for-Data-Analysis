/**
 * app/api/access/route.ts — checks the access code and sets the cookie.
 *   POST   { code }  → 200 + cookie on a match, 401 otherwise
 *   DELETE           → signs out (clears the cookie)
 */

import { NextRequest, NextResponse } from "next/server";
import {
  ACCESS_COOKIE,
  ACCESS_COOKIE_MAX_AGE,
  configuredAccessCode,
  gateMode,
  safeEqual,
  tokenFor,
} from "@/lib/accessGate";

export const runtime = "nodejs";

// Slows down guessing: failed attempts per IP within a 15-minute window.
const FAIL_WINDOW_MS = 15 * 60 * 1000;
const MAX_FAILS = 10;
const failures = new Map<string, { count: number; first: number }>();

function clientIp(request: NextRequest): string {
  // The caller controls the FIRST X-Forwarded-For entry, so trusting it lets anyone reset the
  // lockout below by sending a fresh fake address per guess. x-vercel-forwarded-for / x-real-ip
  // are only trustworthy on Vercel, which overwrites them at its edge; on any other host a
  // caller could send them freely, so they are ignored unless VERCEL is set. Everywhere else
  // use the LAST x-forwarded-for hop, which is appended by the nearest proxy.
  const h = request.headers;
  if (process.env.VERCEL) {
    const platform = h.get("x-vercel-forwarded-for")?.split(",")[0]?.trim() || h.get("x-real-ip")?.trim();
    if (platform) return platform;
  }
  const hops = (h.get("x-forwarded-for") ?? "").split(",").map((s) => s.trim()).filter(Boolean);
  return hops[hops.length - 1] || "unknown";
}

export async function POST(request: NextRequest) {
  if (gateMode() === "off") return NextResponse.json({ success: true });
  if (gateMode() === "misconfigured") {
    return NextResponse.json({ success: false, error: "ACCESS_CODE is not configured." }, { status: 503 });
  }

  const ip = clientIp(request);
  const now = Date.now();
  const record = failures.get(ip);
  if (record && now - record.first < FAIL_WINDOW_MS && record.count >= MAX_FAILS) {
    return NextResponse.json(
      { success: false, error: "Too many attempts. Try again in 15 minutes." },
      { status: 429 }
    );
  }

  const body = await request.json().catch(() => null);
  const code = typeof body?.code === "string" ? body.code.trim() : "";
  const [given, expected] = await Promise.all([tokenFor(code), tokenFor(configuredAccessCode())]);

  if (!code || !safeEqual(given, expected)) {
    const fresh = !record || now - record.first >= FAIL_WINDOW_MS;
    failures.set(ip, fresh ? { count: 1, first: now } : { count: record!.count + 1, first: record!.first });
    await new Promise((r) => setTimeout(r, 600)); // blunt brute force
    return NextResponse.json({ success: false, error: "That code isn't right." }, { status: 401 });
  }

  failures.delete(ip);
  const response = NextResponse.json({ success: true });
  response.cookies.set(ACCESS_COOKIE, expected, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: ACCESS_COOKIE_MAX_AGE,
  });
  return response;
}

export async function DELETE() {
  const response = NextResponse.json({ success: true });
  response.cookies.set(ACCESS_COOKIE, "", { path: "/", maxAge: 0 });
  return response;
}
