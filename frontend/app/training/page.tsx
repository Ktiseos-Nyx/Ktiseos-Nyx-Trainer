'use client';

import dynamic from 'next/dynamic';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Home, Settings, Loader2 } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';

// 🚀 OPTIMIZATION: Lazy load heavy training components
const TrainingConfig = dynamic(() => import('@/components/training/TrainingConfig'), {
  loading: () => (
    <Card>
      <CardContent className="p-12 flex items-center justify-center">
        <div className="text-center space-y-4">
          <Loader2 className="w-8 h-8 animate-spin mx-auto text-primary" />
          <p className="text-muted-foreground">Loading training configuration...</p>
        </div>
      </CardContent>
    </Card>
  ),
  ssr: false,
});

const TrainingMonitor = dynamic(() => import('@/components/training/TrainingMonitor'), {
  loading: () => (
    <Card>
      <CardContent className="p-12 flex items-center justify-center">
        <div className="text-center space-y-4">
          <Loader2 className="w-8 h-8 animate-spin mx-auto text-primary" />
          <p className="text-muted-foreground">Loading training monitor...</p>
        </div>
      </CardContent>
    </Card>
  ),
  ssr: false,
});

export default function TrainingPage() {
  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'LoRA Training', icon: <Settings className="w-4 h-4" /> },
          ]}
        />

        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent">
            LoRA Training
          </h1>
          <p className="text-xl text-muted-foreground">
            Configure and monitor your training runs
          </p>
        </div>

        {/* Main Content */}
        <div className="space-y-8">
          {/* Training Configuration */}
          <TrainingConfig />

          {/* Training Monitor */}
          <TrainingMonitor />
        </div>
      </div>
    </div>
  );
}
