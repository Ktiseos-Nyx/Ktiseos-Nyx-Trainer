import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { Toaster } from "@/components/ui/sonner";
import { Navbar } from "@/components/blocks/navigation/navbar";
import { Footer } from "@/components/blocks/navigation/footer";
import { MeshGradientBackground } from "@/components/ui/mesh-gradient";

export const metadata: Metadata = {
  title: "Ktiseos-Nyx LoRA Trainer",
  description: "Professional LoRA training with modern web UI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased flex flex-col min-h-screen">
        <div className="fixed inset-0 -z-10">
          <MeshGradientBackground />
        </div>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem={true}
          disableTransitionOnChange
        >
          <Navbar />
          <main className="flex-1 flex flex-col">{children}</main>
          <Footer />
          <Toaster richColors position="bottom-right" />
        </ThemeProvider>
      </body>
    </html>
  );
}
