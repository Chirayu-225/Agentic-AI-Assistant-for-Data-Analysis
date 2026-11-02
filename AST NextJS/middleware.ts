/**
 * middleware.ts — access-code gate for every page and API route.
 * See lib/accessGate.ts for how it works.
 *
 * Pages without a valid cookie redirect to /access; API calls get a 401
 * JSON error instead (a redirect would just confuse fetch()).
 */

import { NextResponse, type NextRequest } from "next/server";
import { ACCESS_COOKIE, configuredAccessCode, gateMode, safeEqual, tokenFor } from "@/lib/accessGate";

export async function middleware(request: NextRequest) {
  const mode = gateMode();
  if (mode === "off") return NextResponse.next();

  const { pathname, search } = request.nextUrl;
  const isApi = pathname.startsWith("/api/");

  if (mode === "misconfigured") {
    const message = "Access is locked: ACCESS_CODE is not configured on the server.";
    return isApi
      ? NextResponse.json({ success: false, error: message }, { status: 503 })
      : new NextResponse(message, { status: 503 });
  }

  const cookie = request.cookies.get(ACCESS_COOKIE)?.value ?? "";
  const expected = await tokenFor(configuredAccessCode());
  if (cookie && safeEqual(cookie, expected)) return NextResponse.next();

  if (isApi) {
    return NextResponse.json(
      { success: false, error: "Access code required. Reload the page to sign in." },
      { status: 401 }
    );
  }
  const url = request.nextUrl.clone();
  url.pathname = "/access";
  url.search = `?next=${encodeURIComponent(pathname + search)}`;
  return NextResponse.redirect(url);
}

export const config = {
  // Everything except the gate itself, Next.js internals, the tab icon and static files —
  // and /api/upload. Next.js copies a request body for middleware with a
  // 10 MB limit (middlewareClientMaxBodySize) and TRUNCATES anything larger,
  // so an 18 MB .xlsx reached the backend cut in half ("File is not a zip
  // file") and a big CSV would silently lose rows. The upload route checks
  // the access cookie itself instead (app/api/upload/route.ts).
  matcher: ["/((?!access|api/access|api/upload|_next/static|_next/image|favicon.ico|icon.svg).*)"],
};
