'use client';

import { Home, GitMerge } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { LoRAToCheckpointTab } from '@/components/merge/LoRAToCheckpoint';

export default function BakePage() {
  return (
    <div className="min-h-screen py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        <Breadcrumbs items={[
          { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
          { label: 'Merge Tools', href: '/merge', icon: <GitMerge className="w-4 h-4" /> },
          { label: 'LoRA → Checkpoint' },
        ]} />
        <div className="mb-8">
          <h1 className="text-4xl md:text-5xl font-bold mb-2 bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">
            LoRA → Checkpoint
          </h1>
          <p className="text-muted-foreground">Bake one or more LoRAs into a base checkpoint</p>
        </div>
        <LoRAToCheckpointTab />
      </div>
    </div>
  );
}
