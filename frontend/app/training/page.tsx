import TrainingConfig from '@/components/training/TrainingConfig';
import TrainingMonitor from '@/components/training/TrainingMonitor';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Home, Settings } from 'lucide-react';

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
