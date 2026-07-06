'use client';

import dynamic from 'next/dynamic';
import { Sparkles, Settings } from 'lucide-react';
// 🚀 OPTIMIZATION: Lazy load hero animation (saves 6.4MB GSAP bundle)
const HeroAnimated = dynamic(() => import('@/components/blocks/hero/hero-animated').then(mod => ({ default: mod.HeroAnimated })), {
  ssr: false,
  loading: () => (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-background to-background/80">
      <div className="text-center space-y-4">
        <div className="animate-pulse text-6xl">✨</div>
        <p className="text-muted-foreground">Loading...</p>
      </div>
    </div>
  ),
});

export default function Home() {
  return (
    <div className="flex-1 flex flex-col">
    <HeroAnimated
      title="Ktiseos Nyx TRAINER"
      subtitle="Model Training Made Beautiful"
      description="A powerful, web-based training ecosystem for creating AI models. Upload datasets, configure training parameters, and monitor progress—all in one beautiful interface."
      features={[
        { icon: '🎯', label: 'Easy Dataset Management' },
        { icon: '⚡', label: 'Fast Training' },
        { icon: '📊', label: 'Real-time Monitoring' },
        { icon: '🎨', label: 'Beautiful UI' },
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
      helperText="Optimized for VastAI & Remote containers and local development"
      theme="purple-blue"
    />
    </div>
  );
}
