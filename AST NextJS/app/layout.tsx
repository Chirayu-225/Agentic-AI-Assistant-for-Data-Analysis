import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Analyst Assistant",
  description: "AI-powered data intelligence — secure analytics terminal",
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
