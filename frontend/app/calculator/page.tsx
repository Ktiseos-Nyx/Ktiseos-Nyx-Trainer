'use client';

import { useState, useEffect } from 'react';
import { utilitiesAPI, CalculatorResponse, DatasetInfo } from '@/lib/api';
import { Calculator, FolderOpen, Info, Zap, Clock, TrendingUp, Home } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';

export default function CalculatorPage() {
  const [datasetPath, setDatasetPath] = useState('');
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(1);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [result, setResult] = useState<CalculatorResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load available datasets
  useEffect(() => {
    const loadDatasets = async () => {
      try {
        const data = await utilitiesAPI.browseDatasets();
        setDatasets(data.datasets || []);

        // Auto-select most recent if available
        if (data.datasets && data.datasets.length > 0) {
          setDatasetPath(data.datasets[0].path);
        }
      } catch (err) {
        console.error('Failed to load datasets:', err);
      }
    };
    loadDatasets();
  }, []);

  const handleCalculate = async () => {
    if (!datasetPath) {
      setError('Please select a dataset');
      return;
    }

    if (batchSize <= 0) {
      setError('Batch size must be greater than 0');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await utilitiesAPI.calculateSteps({
        dataset_path: datasetPath,
        epochs,
        batch_size: batchSize,
      });

      setResult(response);
    } catch (err: any) {
      setError(err.message || 'Calculation failed');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Step Calculator', icon: <Calculator className="w-4 h-4" /> },
          ]}
        />

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 text-foreground">
            LoRA Step Calculator
          </h1>
          <p className="text-xl text-muted-foreground">
            Calculate optimal training steps with Kohya-compatible logic
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Left: Input Form */}
          <div className="bg-card backdrop-blur-sm border border-border rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold mb-4 text-foreground">Training Parameters</h2>

            {/* Dataset Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-muted-foreground mb-2">
                Dataset Path
              </label>

              {datasets.length > 0 ? (
                <select
                  value={datasetPath}
                  onChange={(e) => setDatasetPath(e.target.value)}
                  className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 mb-2"
                >
                  {datasets.map((dataset) => (
                    <option key={dataset.path} value={dataset.path}>
                      {dataset.name} ({dataset.image_count} images, {dataset.repeats}x repeats)
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={datasetPath}
                  onChange={(e) => setDatasetPath(e.target.value)}
                  className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                  placeholder="datasets/10_character_name"
                />
              )}

              <div className="bg-muted/50 border border-border rounded-lg p-3 text-sm">
                <p className="font-medium text-foreground mb-1">
                  Kohya Format
                </p>
                <p className="text-muted-foreground">
                  Auto-detects repeat counts from folder names (e.g., "10_character_name" → 10 repeats)
                </p>
              </div>
            </div>

            {/* Epochs */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-muted-foreground mb-2">
                Epochs
              </label>
              <input
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value) || 0)}
                className="w-full px-3 py-2 border border-input bg-input text-foreground rounded-lg focus:ring-2 focus:ring-blue-500"
                min="1"
              />
            </div>

            {/* Batch Size */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-muted-foreground mb-2">
                Batch Size
              </label>
              <input
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value) || 0)}
                className="w-full px-3 py-2 border border-input bg-input text-foreground rounded-lg focus:ring-2 focus:ring-blue-500"
                min="1"
              />
            </div>

            {/* Calculate Button */}
            <button
              onClick={handleCalculate}
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-blue-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              <Calculator className="w-5 h-5" />
              {loading ? 'Calculating...' : 'Calculate Steps'}
            </button>

            {/* Error Display */}
            {error && (
              <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
                {error}
              </div>
            )}

            {/* Info Box */}
            <div className="mt-6 bg-muted/50 border border-border rounded-lg p-4">
              <p className="text-sm font-medium text-foreground mb-2">Formula</p>
              <code className="text-xs bg-background text-foreground px-2 py-1 rounded border border-border">
                (Images × Repeats × Epochs) ÷ Batch Size = Steps
              </code>
            </div>
          </div>

          {/* Right: Results */}
          <div className="bg-card rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold mb-4 text-foreground">Results</h2>

            {result ? (
              <div className="space-y-4">
                {/* Total Steps - Big Display */}
                <div className="bg-gradient-to-r from-green-500 to-emerald-600 rounded-lg p-6 text-white text-center">
                  <div className="flex items-center justify-center gap-2 mb-2">
                    <Zap className="w-6 h-6" />
                    <span className="text-sm font-medium">Total Steps</span>
                  </div>
                  <div className="text-5xl font-bold">{result.total_steps.toLocaleString()}</div>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                    <div className="text-sm text-blue-600 dark:text-blue-400 mb-1">Images</div>
                    <div className="text-2xl font-bold">{result.images}</div>
                  </div>

                  <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                    <div className="text-sm text-purple-600 dark:text-purple-400 mb-1">Repeats</div>
                    <div className="text-2xl font-bold">{result.repeats}x</div>
                  </div>

                  <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
                    <div className="text-sm text-orange-600 dark:text-orange-400 mb-1">Epochs</div>
                    <div className="text-2xl font-bold">{result.epochs}</div>
                  </div>

                  <div className="bg-pink-50 dark:bg-pink-900/20 rounded-lg p-4">
                    <div className="text-sm text-pink-600 dark:text-pink-400 mb-1">Batch Size</div>
                    <div className="text-2xl font-bold">{result.batch_size}</div>
                  </div>
                </div>

                {/* Caption */}
                {result.caption && (
                  <div className="bg-card rounded-lg p-4 border border-border">
                    <div className="text-sm font-medium text-muted-foreground mb-1">
                      Detected Caption
                    </div>
                    <div className="font-mono text-sm text-foreground">{result.caption}</div>
                  </div>
                )}

                {/* Time Estimates */}
                <div className="border border-border rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Clock className="w-5 h-5 text-foreground" />
                    <span className="font-semibold text-foreground">Time Estimates (approximate)</span>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">GPU Rental (faster):</span>
                      <span className="font-medium">{result.time_estimate_min.toFixed(1)} min</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Home GPU (slower):</span>
                      <span className="font-medium">{result.time_estimate_max.toFixed(1)} min</span>
                    </div>
                  </div>
                </div>

                {/* Recommendation */}
                <div className={`rounded-lg p-4 ${
                  result.recommendation.includes('✅')
                    ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800'
                    : 'bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800'
                }`}>
                  <div className="flex items-start gap-2">
                    <TrendingUp className="w-5 h-5 mt-0.5 flex-shrink-0" />
                    <div>
                      <div className="font-semibold mb-1">Training Analysis</div>
                      <div className="text-sm">{result.recommendation}</div>
                    </div>
                  </div>
                </div>

                {/* Dataset Path */}
                <div className="text-xs text-muted-foreground font-mono bg-card rounded p-2 border border-border">
                  {result.dataset_path}
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <Calculator className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                <p className="text-muted-foreground">
                  Enter your parameters and click Calculate
                </p>
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
