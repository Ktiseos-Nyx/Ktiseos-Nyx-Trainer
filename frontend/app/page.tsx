'use client';

import { Sparkles, Settings, AlertTriangle } from 'lucide-react';
import { HeroAnimated } from '@/components/blocks/hero/hero-animated';
import { FeatureGrid } from '@/components/blocks/features/feature-grid';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

export default function Home() {
  return (
    <div className="min-h-screen">
      {/* Beta Warning Banner */}
      <div className="container mx-auto px-4 pt-6">
        <Alert variant="default" className="border-yellow-500/50 bg-yellow-500/10 text-yellow-200">
          <AlertTriangle className="h-4 w-4 text-yellow-500" />
          <AlertTitle className="text-yellow-100">Beta - In Active Development</AlertTitle>
          <AlertDescription className="text-yellow-200/80">
            This project is currently in beta. Features may not work as expected, and breaking changes may occur.
            Please report issues on{' '}
            <a
              href="https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues"
              target="_blank"
              rel="noopener noreferrer"
              className="underline hover:text-yellow-100"
            >
              GitHub
            </a>.
          </AlertDescription>
        </Alert>
      </div>

      <HeroAnimated
        title="Ktiseos-Nyx"
        subtitle="LoRA Trainer"
        description="Professional-grade LoRA training with a modern, intuitive web interface"
        features={[
          { icon: '🎯', label: 'Kohya SD-Scripts' },
          { icon: '⚡', label: 'Real-time Monitoring' },
          { icon: '🎨', label: 'Auto-tagging' },
        ]}
        ctas={[
          {
            label: 'Get Started',
            href: '/dashboard',
            variant: 'primary',
            icon: Sparkles,
          },
          {
            label: 'Settings',
            href: '/settings',
            variant: 'secondary',
            icon: Settings,
          },
        ]}
        helperText="Or just start via the navigation bar!"
        theme="purple-blue"
      />

      <FeatureGrid
        title="Everything You Need"
        features={[
          {
            emoji: '🎯',
            title: 'Easy Setup',
            description: 'No complex configuration required. Upload datasets, configure training, and go.',
          },
          {
            emoji: '⚡',
            title: 'Real-time Monitoring',
            description: 'Watch your training progress with live updates via WebSockets.',
          },
          {
            emoji: '🎨',
            title: 'Powerful Tools',
            description: 'WD14 tagging, step calculator, LoRA resizing, and HuggingFace integration.',
          },
        ]}
        ctaLabel="Start Training Now →"
        ctaHref="/dashboard"
        columns={3}
      />
    </div>
  );
}
