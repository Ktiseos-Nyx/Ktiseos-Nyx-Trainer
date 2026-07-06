// frontend/app/dashboard/page.tsx

'use client';

import Link from 'next/link';
import {
  FolderOpen,
  Settings,
  Upload,
  Calculator,
  Wrench,
  Download,
  Zap,
  BookOpen,
  Info,
  type LucideIcon,
} from 'lucide-react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from '@/components/ui/card';

type Tool = {
  href: string;
  icon: LucideIcon;
  title: string;
  description: string;
  /** The primary entry point — gets a little extra emphasis/glow. */
  primary?: boolean;
};

// Ordered to lead with the core workflow (train → data → models) rather than
// the old alphabetical-ish artifact order.
const tools: Tool[] = [
  { href: '/training', icon: Zap, title: 'Training', description: 'Configure and monitor LoRA training', primary: true },
  { href: '/dataset', icon: Upload, title: 'Dataset', description: 'Upload and prepare training datasets' },
  { href: '/models', icon: Download, title: 'Models & VAEs', description: 'Download base models and VAEs' },
  { href: '/files', icon: FolderOpen, title: 'File Manager', description: 'Browse, upload, and manage your files' },
  { href: '/calculator', icon: Calculator, title: 'Calculator', description: 'Calculate optimal training steps' },
  { href: '/utilities', icon: Wrench, title: 'Utilities', description: 'Manage and upload trained LoRAs' },
  { href: '/settings', icon: Settings, title: 'Settings', description: 'Configure app preferences and defaults' },
  { href: '/docs', icon: BookOpen, title: 'Documentation', description: 'Guides, tutorials, and best practices' },
  { href: '/about', icon: Info, title: 'About', description: 'Learn about the project and credits' },
];

/**
 * Training Dashboard — a launcher grid linking to every tool in the app.
 *
 * Cards are real navigation links (anchor-wrapped) for keyboard accessibility,
 * built on the shared shadcn Card with a single theme accent + a subtle glow on
 * hover/focus. The primary action (Training) carries a touch more emphasis.
 */
export default function Dashboard() {
  return (
    <div className="min-h-screen">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold text-foreground mb-4">Training Dashboard</h1>
          <p className="text-xl text-muted-foreground">Quick access to all your training tools</p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
          {tools.map(({ href, icon: Icon, title, description, primary }) => (
            <Link
              key={href}
              href={href}
              prefetch={false}
              className="group block rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60 focus-visible:ring-offset-2 focus-visible:ring-offset-background"
            >
              <Card
                className={
                  'h-full gap-3 px-6 transition-all hover:-translate-y-0.5 ' +
                  (primary
                    ? 'border-primary/40 bg-primary/5 hover:border-primary/70 hover:shadow-[0_8px_30px_-6px_hsl(var(--primary)/0.40)]'
                    : 'hover:border-primary/50 hover:shadow-[0_6px_24px_-8px_hsl(var(--primary)/0.25)]')
                }
              >
                <Icon className="w-10 h-10 text-primary transition-transform group-hover:scale-110" />
                <div className="space-y-1">
                  <CardTitle>{title}</CardTitle>
                  <CardDescription>{description}</CardDescription>
                </div>
              </Card>
            </Link>
          ))}
        </div>

        <Card className="mt-16 max-w-4xl mx-auto">
          <CardHeader>
            <CardTitle className="text-2xl">Quick Start</CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="list-decimal list-inside space-y-2 text-muted-foreground">
              <li>Download a base model from <strong className="text-primary">Models &amp; VAEs</strong></li>
              <li>Upload your training images via <strong className="text-primary">Dataset</strong></li>
              <li>Auto-tag images with the WD14 tagger</li>
              <li>Configure training parameters in <strong className="text-primary">Training</strong></li>
              <li>Monitor progress in real-time</li>
              <li>Download your trained LoRA model</li>
            </ol>
          </CardContent>
        </Card>

        <div className="mt-8 text-center text-sm text-muted-foreground">
          <p>Powered by Kohya SD-Scripts &bull; Built with Next.js &amp; FastAPI</p>
        </div>
      </div>
    </div>
  );
}
