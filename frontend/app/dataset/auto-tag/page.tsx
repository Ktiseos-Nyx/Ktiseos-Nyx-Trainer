'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { datasetAPI, DatasetInfo } from '@/lib/api';
import { Home, Database, Tag, Zap, Info, ChevronDown, ChevronUp, Terminal } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { GradientCard } from '@/components/effects';

export default function AutoTagPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(true);

  // Tagging state
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [taggerModel, setTaggerModel] = useState('wd14-vit-v2');
  const [threshold, setThreshold] = useState(0.35);
  const [batchSize, setBatchSize] = useState(4);
  const [tagging, setTagging] = useState(false);

  // Log viewer state
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);

  // Load datasets
  const loadDatasets = async () => {
    try {
      setLoading(true);
      const data = await datasetAPI.list();
      setDatasets(data.datasets || []);
      if (data.datasets && data.datasets.length > 0 && !selectedDataset) {
        setSelectedDataset(data.datasets[0].path);
      }
    } catch (err) {
      console.error('Failed to load datasets:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  // Handle tagging
  const handleStartTagging = async () => {
    if (!selectedDataset) {
      alert('Please select a dataset to tag!');
      return;
    }

    setTagging(true);
    setShowLogs(true);
    setLogs([]);

    // Add initial log
    const addLog = (msg: string) => {
      setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
    };

    addLog('🚀 Starting auto-tagging process...');
    addLog(`📁 Dataset: ${selectedDataset}`);
    addLog(`🤖 Model: ${taggerModel}`);
    addLog(`🎯 Threshold: ${threshold}`);
    addLog(`📦 Batch size: ${batchSize}`);

    try {
      addLog('📡 Sending request to backend...');
      await datasetAPI.tag(selectedDataset, taggerModel, threshold);
      addLog('✅ Tagging started! Processing in background...');
      addLog('💡 Tags will be saved as .txt files next to each image');

      // Reload datasets to update tag status
      setTimeout(() => {
        loadDatasets();
        addLog('🔄 Dataset list refreshed');
      }, 2000);
    } catch (err) {
      addLog(`❌ Tagging failed: ${err}`);
      alert(`❌ Tagging failed: ${err}`);
    } finally {
      setTagging(false);
    }
  };

  const selectedDatasetInfo = datasets.find(d => d.path === selectedDataset);

  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Dataset Management', href: '/dataset', icon: <Database className="w-4 h-4" /> },
            { label: 'Auto-Tagging', icon: <Tag className="w-4 h-4" /> },
          ]}
        />

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-pink-400 via-rose-400 to-pink-400 bg-clip-text text-transparent">
            Auto-Tagging
          </h1>
          <p className="text-xl text-muted-foreground mt-4">
            Automatically generate captions for your training images using WD14 tagger models
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Left: Configuration */}
          <GradientCard variant="watermelon" intensity="subtle">
            <div className="p-6 space-y-6">
              <h2 className="text-2xl font-bold text-foreground flex items-center gap-2">
                <Zap className="w-6 h-6 text-pink-600 dark:text-pink-400" />
                Configuration
              </h2>

              {/* Dataset Selection */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Select Dataset
                </label>
                {loading ? (
                  <div className="text-sm text-muted-foreground">Loading datasets...</div>
                ) : datasets.length === 0 ? (
                  <div className="text-sm text-muted-foreground">
                    No datasets found.{' '}
                    <Link href="/dataset" className="text-pink-600 dark:text-pink-400 hover:underline">
                      Upload some images first
                    </Link>
                  </div>
                ) : (
                  <select
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    className="w-full px-4 py-2 bg-input border border-border rounded-lg text-foreground focus:ring-2 focus:ring-pink-500 focus:border-transparent"
                  >
                    {datasets.map((ds) => (
                      <option key={ds.path} value={ds.path}>
                        {ds.name} ({ds.image_count} images)
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {/* Model Selection */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Tagger Model
                </label>
                <select
                  value={taggerModel}
                  onChange={(e) => setTaggerModel(e.target.value)}
                  className="w-full px-4 py-2 bg-input border border-border rounded-lg text-foreground focus:ring-2 focus:ring-pink-500 focus:border-transparent"
                >
                  <option value="wd14-vit-v2">WD14 ViT v2 (Recommended)</option>
                  <option value="wd14-vit-v3">WD14 ViT v3 (Latest)</option>
                  <option value="wd14-convnext-v2">WD14 ConvNext v2</option>
                  <option value="wd14-convnext-v3">WD14 ConvNext v3</option>
                  <option value="wd14-swinv2-v2">WD14 SwinV2 v2</option>
                  <option value="wd14-swinv2-v3">WD14 SwinV2 v3</option>
                </select>
                <p className="text-xs text-muted-foreground mt-2">
                  v2 models are well-tested and reliable, v3 models are newer with improved accuracy
                </p>
              </div>

              {/* Threshold */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Confidence Threshold: {threshold.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-pink-500"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>More tags (0.1)</span>
                  <span>Fewer tags (0.9)</span>
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Lower threshold = more tags but less confident. Higher = only highly confident tags.
                </p>
              </div>

              {/* Batch Size */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Batch Size: {batchSize}
                </label>
                <input
                  type="range"
                  min="1"
                  max="16"
                  step="1"
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-pink-500"
                />
                <p className="text-xs text-muted-foreground mt-2">
                  Higher batch size = faster processing (requires more GPU memory)
                </p>
              </div>

              {/* Start Button */}
              <button
                onClick={handleStartTagging}
                disabled={tagging || !selectedDataset || datasets.length === 0}
                className="w-full px-6 py-3 bg-gradient-to-r from-pink-500 to-rose-500 hover:from-pink-600 hover:to-rose-600 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed text-white rounded-lg font-semibold transition-all shadow-lg flex items-center justify-center gap-2"
              >
                <Tag className="w-5 h-5" />
                {tagging ? 'Tagging in Progress...' : 'Start Auto-Tagging'}
              </button>
            </div>
          </GradientCard>

          {/* Right: Info & Dataset Details */}
          <div className="space-y-6">
            {/* How it Works */}
            <GradientCard variant="ocean" intensity="subtle">
              <div className="p-6">
                <h3 className="text-xl font-bold text-foreground mb-4 flex items-center gap-2">
                  <Info className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
                  How Auto-Tagging Works
                </h3>
                <ul className="space-y-3 text-sm text-foreground">
                  <li className="flex items-start gap-2">
                    <span className="text-cyan-400 mt-0.5">1.</span>
                    <span>WD14 tagger analyzes each image using deep learning</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-cyan-400 mt-0.5">2.</span>
                    <span>Generates descriptive tags (characters, composition, art style, quality)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-cyan-400 mt-0.5">3.</span>
                    <span>Saves tags as .txt files next to each image (Kohya format)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-cyan-400 mt-0.5">4.</span>
                    <span>You can edit tags afterwards in the Tag Editor</span>
                  </li>
                </ul>

                <div className="mt-4 pt-4 border-t border-border">
                  <p className="text-xs text-muted-foreground">
                    <strong>Tip:</strong> Lower threshold (0.2-0.3) gives more detailed tags.
                    Higher threshold (0.5-0.7) gives only the most confident tags.
                  </p>
                </div>
              </div>
            </GradientCard>

            {/* Selected Dataset Info */}
            {selectedDatasetInfo && (
              <GradientCard variant="shadow" intensity="subtle">
                <div className="p-6">
                  <h3 className="text-xl font-bold text-foreground mb-4">Selected Dataset</h3>
                  <div className="space-y-2 text-sm text-foreground">
                    <div className="flex items-center gap-2">
                      <span className="text-orange-400">📁</span>
                      <span className="font-medium">{selectedDatasetInfo.name}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-orange-400">🖼️</span>
                      <span>{selectedDatasetInfo.image_count} images</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {selectedDatasetInfo.tags_present ? (
                        <>
                          <span className="text-green-400">✓</span>
                          <span className="text-green-400">Already has tags (will be overwritten)</span>
                        </>
                      ) : (
                        <>
                          <span className="text-yellow-400">⚠</span>
                          <span className="text-yellow-400">No tags yet</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </GradientCard>
            )}
          </div>
        </div>

        {/* Logs Section */}
        {showLogs && (
          <div className="mt-6">
            <div className="bg-card border border-border rounded-lg overflow-hidden">
              <button
                onClick={() => setShowLogs(!showLogs)}
                className="w-full px-6 py-4 flex items-center justify-between bg-accent/50 hover:bg-accent transition-colors"
              >
                <div className="flex items-center gap-2 text-foreground font-semibold">
                  <Terminal className="w-5 h-5" />
                  Process Logs
                </div>
                {showLogs ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>

              {showLogs && (
                <div className="p-4 bg-black/50 font-mono text-sm text-green-400 max-h-96 overflow-y-auto">
                  {logs.length === 0 ? (
                    <div className="text-muted-foreground">No logs yet...</div>
                  ) : (
                    logs.map((log, idx) => (
                      <div key={idx} className="py-1">
                        {log}
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
