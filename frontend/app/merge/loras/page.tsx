'use client';

import { Home, Puzzle } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { MergeLoRAsTab } from '@/components/merge/MergeLoRAs';

export default function MergeLoRAsPage() {
  return (
    <div className="min-h-screen py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        <Breadcrumbs items={[
          { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
          { label: 'Merge Tools', href: '/merge', icon: <Puzzle className="w-4 h-4" /> },
          { label: 'Merge LoRAs' },
        ]} />
        <div className="mb-8">
          <h1 className="text-4xl md:text-5xl font-bold mb-2 bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent">
            Merge LoRAs
          </h1>
          <p className="text-muted-foreground">Combine multiple LoRAs into a single LoRA file</p>
        </div>
        <MergeLoRAsTab />
      </div>
    </div>
  );
}
