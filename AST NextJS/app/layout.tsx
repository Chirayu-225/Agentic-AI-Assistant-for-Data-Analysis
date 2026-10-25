import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Analyst Assistant",
  description: "Verified answers from finance and operations spreadsheets — every number computed, every computation inspectable.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-sans">{children}</body>
    </html>
  );
}
