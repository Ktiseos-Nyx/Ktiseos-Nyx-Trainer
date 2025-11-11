import TrainingConfig from '@/components/training/TrainingConfig';
import TrainingMonitor from '@/components/training/TrainingMonitor';

export default function TrainingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-50 dark:from-gray-900 dark:to-gray-800 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            LoRA Training
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Configure and monitor your training runs
          </p>
        </div>

        {/* Two Column Layout */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Left: Configuration */}
          <div>
            <TrainingConfig />
          </div>

          {/* Right: Monitor */}
          <div>
            <TrainingMonitor />
          </div>
        </div>
      </div>
    </div>
  );
}
