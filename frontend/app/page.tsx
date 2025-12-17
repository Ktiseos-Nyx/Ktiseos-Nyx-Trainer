'use client';

import { HeroAnimated } from '@/components/blocks/hero/hero-animated';
import { Sparkles, Settings } from 'lucide-react';

export default function Home() {
  return (
    <HeroAnimated
      title="KNX TRAINER"
      subtitle="LoRA Training Made Simple"
      description="A powerful, web-based training environment for creating LoRA models. Upload datasets, configure training parameters, and monitor progress—all in one beautiful interface."
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
      helperText="Optimized for VastAI containers and local development"
      theme="purple-blue"
    />
  );
}
