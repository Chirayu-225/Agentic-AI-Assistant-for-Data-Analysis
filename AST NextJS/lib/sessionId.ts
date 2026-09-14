/**
 * lib/sessionId.ts — a stand-in for real auth.
 *
 * Mirrors the same idea as main.py's st.session_state.session_id: a
 * stable per-browser identifier used to namespace uploads/queries so
 * one user's data can't collide with another's, even before real
 * accounts exist. Once auth is added, replace every call site of
 * getOrCreateSessionId() with the authenticated user's real ID instead —
 * this is explicitly a placeholder, not a security boundary on its own
 * (a cleared localStorage or a different browser just gets a new
 * identity, same as an anonymous session would).
 */

const STORAGE_KEY = "analyst_assistant_session_id";

export function getOrCreateSessionId(): string {
  if (typeof window === "undefined") return ""; // server-side render guard

  let id = window.localStorage.getItem(STORAGE_KEY);
  if (!id) {
    id = crypto.randomUUID();
    window.localStorage.setItem(STORAGE_KEY, id);
  }
  return id;
}
