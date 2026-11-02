/**
 * lib/accessGate.ts — shared pieces of the access-code gate.
 *
 * The gate keeps strangers from using the deployed app (and spending the
 * Groq quota). The visitor enters ACCESS_CODE once on /access; the server
 * sets an httpOnly cookie holding a hash of the code, and middleware.ts
 * checks that cookie on every page and API request. Changing ACCESS_CODE
 * signs everyone out, because old cookies no longer match.
 *
 * Uses Web Crypto so the same code runs in middleware (Edge runtime) and
 * in route handlers (Node runtime).
 */

export const ACCESS_COOKIE = "ast_access";
export const ACCESS_COOKIE_MAX_AGE = 60 * 60 * 24 * 7; // 7 days

export function configuredAccessCode(): string {
  return (process.env.ACCESS_CODE ?? "").trim();
}

/** Gate is OFF only in local development with no code set. In production
 * a missing ACCESS_CODE fails closed, so forgetting it can't leave a
 * deployment open to everyone. */
export function gateMode(): "off" | "on" | "misconfigured" {
  if (configuredAccessCode()) return "on";
  return process.env.NODE_ENV === "production" ? "misconfigured" : "off";
}

export async function tokenFor(code: string): Promise<string> {
  // Salted with the backend secret when present, so the cookie value
  // can't be precomputed from a guessed code alone.
  const salt = process.env.PYTHON_SERVICE_SECRET ?? "";
  const data = new TextEncoder().encode(`ast-access-v1:${salt}:${code}`);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/** Constant-time string comparison (avoids leaking how much matched). */
export function safeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

/** For route handlers the middleware deliberately skips (see middleware.ts
 * matcher). Returns null when the request may proceed, or an error message
 * plus status when it may not. */
export async function checkAccess(cookieValue: string | undefined): Promise<{ error: string; status: number } | null> {
  const mode = gateMode();
  if (mode === "off") return null;
  if (mode === "misconfigured") {
    return { error: "Access is locked: ACCESS_CODE is not configured on the server.", status: 503 };
  }
  const expected = await tokenFor(configuredAccessCode());
  if (cookieValue && safeEqual(cookieValue, expected)) return null;
  return { error: "Access code required. Reload the page to sign in.", status: 401 };
}
