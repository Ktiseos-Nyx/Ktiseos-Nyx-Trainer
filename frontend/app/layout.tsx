import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { Toaster } from "@/components/ui/sonner";
import { Navbar } from "@/components/blocks/navigation/navbar";

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
      <head>
        {/* Pre-hydration: apply accent color before React mounts to prevent flash */}
        <script
          dangerouslySetInnerHTML={{
            __html: `try{var a=localStorage.getItem('app-accent');if(a&&a!=='zinc'){document.documentElement.setAttribute('data-accent',a)}}catch(e){}`,
          }}
        />
      </head>
      <body className="antialiased flex flex-col min-h-screen">
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem={true}
          disableTransitionOnChange
        >
          <Navbar />
          <main className="flex-1">{children}</main>
          <Toaster richColors position="bottom-right" />
        </ThemeProvider>
      </body>
    </html>
  );
}
