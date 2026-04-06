import CheckpointTrainingConfig from '@/components/checkpoint/CheckpointTrainingConfig';
import TrainingMonitor from '@/components/training/TrainingMonitor';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Home, Cpu, AlertTriangle, ChevronRight } from 'lucide-react';
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

        {/* Hardware Requirements Warning */}
        <Card className="mb-8 border-orange-500/50 bg-orange-500/10">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-orange-600 dark:text-orange-400">
              <AlertTriangle className="w-5 h-5" />
              High-End Hardware Required
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <p className="text-muted-foreground">
              Full checkpoint training requires significantly more VRAM and storage than LoRA training.
              Typical requirements: <strong>24GB+ VRAM for SDXL, 40GB+ for Flux/SD3</strong>, and
              <strong> 50-100GB storage</strong> for intermediate saves.
            </p>
            <p className="text-muted-foreground">
              💡 <strong>Tip:</strong> For most use cases, <a href="/training" className="text-primary hover:underline">LoRA training</a> is
              faster, cheaper, and requires less VRAM while producing high-quality results.
            </p>
            <p className="text-sm mt-3">
              <a href="/docs" className="text-primary hover:underline inline-flex items-center gap-1">
                📖 View detailed hardware requirements in Documentation
                <ChevronRight className="w-4 h-4" />
              </a>
            </p>
          </CardContent>
        </Card>

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
