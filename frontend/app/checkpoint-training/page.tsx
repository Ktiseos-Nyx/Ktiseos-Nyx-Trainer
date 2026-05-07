import CheckpointTrainingConfig from '@/components/checkpoint/CheckpointTrainingConfig';
import TrainingMonitor from '@/components/training/TrainingMonitor';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Home, Cpu } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

export default function CheckpointTrainingPage() {
  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Checkpoint Training', icon: <Cpu className="w-4 h-4" /> },
          ]}
        />

        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-orange-400 via-red-400 to-orange-400 bg-clip-text text-transparent">
            Full Model Fine-tuning
          </h1>
          <p className="text-xl text-muted-foreground">
            Train entire checkpoint models from scratch or fine-tune base models
          </p>
        </div>


        {/* Main Content */}
        <div className="space-y-8">
          {/* Checkpoint Training Configuration */}
          <CheckpointTrainingConfig />

          {/* Training Monitor */}
          <TrainingMonitor />
        </div>
      </div>
    </div>
  );
}
