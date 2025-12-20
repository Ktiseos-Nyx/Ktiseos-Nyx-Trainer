'use client';

import { HeroAnimated } from '@/components/blocks/hero/hero-animated';
import { Sparkles, Settings } from 'lucide-react';

export default function Home() {
  return (
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
  );
}
