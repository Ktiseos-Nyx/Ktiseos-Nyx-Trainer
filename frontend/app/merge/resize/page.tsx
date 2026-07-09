'use client';

import { Home, Shrink } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { ResizeLoRATab } from '@/components/merge/ResizeLoRA';

export default function ResizePage() {
  return (
    <div className="min-h-screen py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        <Breadcrumbs items={[
          { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
          { label: 'Merge Tools', href: '/merge', icon: <Shrink className="w-4 h-4" /> },
          { label: 'Resize LoRA' },
        ]} />
        <div className="mb-8">
          <h1 className="text-4xl md:text-5xl font-bold mb-2 bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
            Resize LoRA
          </h1>
          <p className="text-muted-foreground">Change the rank of a LoRA via SVD</p>
        </div>
        <ResizeLoRATab />
      </div>
    </div>
  );
}
