'use client';

import { Sparkles } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { GradientCard } from '@/components/effects';

export default function AboutPage() {
  return (
    <main className="container mx-auto px-4 py-8">
      <Breadcrumbs />
      <div className="max-w-3xl mx-auto text-center mt-20">
        <div className="flex items-center justify-center gap-3 mb-6">
          <Sparkles className="w-8 h-8 text-purple-600" />
          <h1 className="text-3xl font-bold">Ecosystem</h1>
        </div>
        <GradientCard className="p-8">
          <p className="text-lg text-muted-foreground">
            A web-based LoRA training environment.
          </p>
          <p className="text-sm text-muted-foreground mt-4">
            Built with FastAPI + Next.js, powered by Kohya SS.
          </p>
        </GradientCard>
      </div>
    </main>
  );
}
