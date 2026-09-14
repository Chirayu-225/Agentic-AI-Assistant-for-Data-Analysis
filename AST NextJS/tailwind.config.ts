import type { Config } from "tailwindcss";

// Colors lifted directly from the original app's .streamlit/config.toml,
// so the Next.js port keeps the same visual identity rather than
// introducing a new one.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        base: "#0A0D14",
        panel: "#11151F",
        border: "#242B3D",
        accent: "#2DD4BF",
        text: "#DDE1E8",
        muted: "#7C8698",
      },
      fontFamily: {
        mono: ["JetBrains Mono", "monospace"],
        sans: ["Inter", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
