'use client';

import dynamic from 'next/dynamic';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Home, Cpu, Loader2 } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';

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

export default function CheckpointTrainingPage() {
  return (
    <div className="min-h-screen py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Checkpoint Training', icon: <Cpu className="w-4 h-4" /> },
          ]}
        />

        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-orange-400 via-red-400 to-orange-400 bg-clip-text text-transparent">
            Full Model Fine-tuning
          </h1>
          <p className="text-xl text-muted-foreground">
            Train entire checkpoint models from scratch or fine-tune base models
          </p>
        </div>

        <div className="space-y-8">
          <TrainingConfig trainingType="checkpoint" />
          <TrainingMonitor />
        </div>
      </div>
    </div>
  );
}
