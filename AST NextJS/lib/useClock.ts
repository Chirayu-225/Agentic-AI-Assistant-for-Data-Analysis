"use client";

import { useEffect, useState } from "react";

/**
 * Live-updating HH:MM:SS clock, client-side only (avoids SSR hydration
 * mismatch — the server and browser would otherwise render two different
 * timestamps for the same initial render).
 */
export function useClock(): string {
  const [time, setTime] = useState<string | null>(null);

  useEffect(() => {
    const format = () =>
      new Date().toLocaleTimeString("en-US", {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });

    setTime(format());
    const id = setInterval(() => setTime(format()), 1000);
    return () => clearInterval(id);
  }, []);

  return time ?? "--:--:--";
}
