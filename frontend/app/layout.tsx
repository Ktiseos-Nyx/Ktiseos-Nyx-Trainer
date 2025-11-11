import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ktiseos-Nyx LoRA Trainer",
  description: "Professional LoRA training with modern web UI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.Node;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
