'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Home, Database } from 'lucide-react';
import DatasetUploader from '@/components/DatasetUploader';
import Breadcrumbs from '@/components/Breadcrumbs';
import { datasetAPI } from '@/lib/api';
import { FolderOpen, Image as ImageIcon, Tag, Trash2, RefreshCw, ChevronDown, ChevronUp, Terminal } from 'lucide-react';
import { GradientCard } from '@/components/effects';

interface Dataset {
  name: string;
  path: string;
  image_count: number;
  tags_present: boolean;
}

export default function DatasetPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);

  // Tagging state
  const [selectedDatasetForTagging, setSelectedDatasetForTagging] = useState<string>('');
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
    } catch (err) {
      console.error('Failed to load datasets:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  // Delete dataset
  const handleDelete = async (name: string) => {
    if (!confirm(`Delete dataset "${name}"? This cannot be undone!`)) {
      return;
    }

    try {
      await fetch(`/api/dataset/${name}`, { method: 'DELETE' });
      loadDatasets();
    } catch (err) {
      alert(`Failed to delete: ${err}`);
    }
  };

  // Auto-select first dataset when datasets load
  useEffect(() => {
    if (datasets.length > 0 && !selectedDatasetForTagging) {
      setSelectedDatasetForTagging(datasets[0].path);
    }
  }, [datasets]);

  // Handle tagging
  const handleStartTagging = async () => {
    if (!selectedDatasetForTagging) {
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
    addLog(`📁 Dataset: ${selectedDatasetForTagging}`);
    addLog(`🤖 Model: ${taggerModel}`);
    addLog(`🎯 Threshold: ${threshold}`);
    addLog(`📦 Batch size: ${batchSize}`);

    try {
      addLog('📡 Sending request to backend...');
      await datasetAPI.tag(selectedDatasetForTagging, taggerModel, threshold);
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

  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Dataset Management', icon: <Database className="w-4 h-4" /> },
          ]}
        />

        {/* Page Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-green-400 via-teal-400 to-green-400 bg-clip-text text-transparent">
                Dataset Management
              </h1>
              <p className="text-xl text-gray-300">
                Upload and manage your training datasets
              </p>
            </div>
            <Link
              href="/dataset/tags"
              className="px-6 py-3 bg-gradient-to-r from-green-500 to-teal-500 hover:from-green-600 hover:to-teal-600 text-white rounded-lg font-semibold transition-all shadow-lg"
            >
              <Tag className="w-5 h-5 inline-block mr-2" />
              Edit Tags
            </Link>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left: Uploader (2/3 width) */}
          <div className="lg:col-span-2">
            <DatasetUploader />
          </div>

          {/* Right: Dataset List (1/3 width) */}
          <GradientCard variant="ocean" intensity="subtle">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-white">Existing Datasets</h2>
                <button
                  onClick={loadDatasets}
                  className="flex items-center gap-1 text-sm text-cyan-400 hover:text-cyan-300 transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  Refresh
                </button>
              </div>

              {loading ? (
                <div className="text-center py-8 text-gray-400">
                  Loading...
                </div>
              ) : datasets.length === 0 ? (
                <div className="text-center py-12">
                  <FolderOpen className="w-16 h-16 mx-auto text-gray-600 mb-4" />
                  <p className="text-gray-400 text-sm">
                    No datasets yet.
                    <br />
                    Upload some images to get started!
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {datasets.map((dataset) => (
                    <div
                      key={dataset.name}
                      className="border border-slate-700 rounded-lg p-4 hover:bg-slate-800/50 transition-colors"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <FolderOpen className="w-5 h-5 text-cyan-400" />
                          <h3 className="font-semibold text-white">{dataset.name}</h3>
                        </div>
                        <button
                          onClick={() => handleDelete(dataset.name)}
                          className="text-red-400 hover:text-red-300 p-1"
                          title="Delete dataset"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>

                      <div className="text-sm text-gray-400 space-y-1">
                        <div className="flex items-center gap-2">
                          <ImageIcon className="w-4 h-4" />
                          <span>{dataset.image_count} images</span>
                        </div>

                        <div className="flex items-center gap-2">
                          <Tag className="w-4 h-4" />
                          <span>
                            {dataset.tags_present ? (
                              <span className="text-green-400">✓ Tagged</span>
                            ) : (
                              <span className="text-yellow-400">No tags</span>
                            )}
                          </span>
                        </div>
                      </div>

                      <div className="mt-3 text-xs text-gray-500 font-mono">
                        {dataset.path}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </GradientCard>
        </div>

        {/* Tagging Configuration */}
        <GradientCard variant="watermelon" intensity="subtle" className="mt-8">
          <div className="p-6">
              <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                <Tag className="w-6 h-6 text-pink-400" />
                Auto-Tagging Configuration
              </h2>
              <p className="text-gray-300 mb-6">
                Automatically caption your images using WD14 tagger models
              </p>

              <div className="grid md:grid-cols-2 gap-6">
                {/* Left: Configuration */}
                <div className="space-y-4">
                  {/* Dataset Selection */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Select Dataset
                    </label>
                    <select
                      value={selectedDatasetForTagging}
                      onChange={(e) => setSelectedDatasetForTagging(e.target.value)}
                      className="w-full px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-pink-500 focus:border-transparent"
                    >
                      {datasets.map((ds) => (
                        <option key={ds.path} value={ds.path}>
                          {ds.name} ({ds.image_count} images)
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Model Selection */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Tagger Model
                    </label>
                    <select
                      value={taggerModel}
                      onChange={(e) => setTaggerModel(e.target.value)}
                      className="w-full px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-pink-500 focus:border-transparent"
                    >
                      <option value="wd14-vit-v2">WD14 ViT v2 (Recommended)</option>
                      <option value="wd14-vit-v3">WD14 ViT v3 (Latest)</option>
                      <option value="wd14-convnext-v2">WD14 ConvNext v2</option>
                      <option value="wd14-convnext-v3">WD14 ConvNext v3</option>
                      <option value="wd14-swinv2-v2">WD14 SwinV2 v2</option>
                      <option value="wd14-swinv2-v3">WD14 SwinV2 v3</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      💡 v2 models are well-tested, v3 models are newer
                    </p>
                  </div>

                  {/* Threshold */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Threshold: {threshold.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.05"
                      value={threshold}
                      onChange={(e) => setThreshold(parseFloat(e.target.value))}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>More tags (0.1)</span>
                      <span>Fewer tags (0.9)</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      💡 Lower = more tags, Higher = only confident tags
                    </p>
                  </div>

                  {/* Batch Size */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Batch Size: {batchSize}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="16"
                      step="1"
                      value={batchSize}
                      onChange={(e) => setBatchSize(parseInt(e.target.value))}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      💡 Higher = faster (if you have GPU memory)
                    </p>
                  </div>
                </div>

                {/* Right: Info & Actions */}
                <div className="space-y-4">
                  {/* Info Box */}
                  <div className="bg-pink-500/10 rounded-lg border border-pink-500/30 p-4">
                    <h4 className="font-semibold text-pink-300 mb-2">How it works</h4>
                    <ul className="text-sm text-gray-300 space-y-2">
                      <li className="flex items-start gap-2">
                        <span className="text-pink-400 mt-0.5">•</span>
                        <span>WD14 tagger analyzes each image</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-pink-400 mt-0.5">•</span>
                        <span>Generates descriptive tags (characters, composition, style)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-pink-400 mt-0.5">•</span>
                        <span>Saves as .txt files next to each image</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-pink-400 mt-0.5">•</span>
                        <span>Edit tags afterwards in Tag Editor</span>
                      </li>
                    </ul>
                  </div>

                  {/* Selected Dataset Info */}
                  {selectedDatasetForTagging && (
                    <div className="bg-slate-800/50 rounded-lg border border-slate-600 p-4">
                      <h4 className="font-semibold text-white mb-2">Selected Dataset</h4>
                      {(() => {
                        const dataset = datasets.find(d => d.path === selectedDatasetForTagging);
                        if (!dataset) return null;
                        return (
                          <div className="text-sm text-gray-300 space-y-1">
                            <div>📁 {dataset.name}</div>
                            <div>🖼️ {dataset.image_count} images</div>
                            <div>
                              {dataset.tags_present ? (
                                <span className="text-green-400">✓ Already has tags</span>
                              ) : (
                                <span className="text-yellow-400">⚠ No tags yet</span>
                              )}
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  )}

                  {/* Start Button */}
                  <button
                    onClick={handleStartTagging}
                    disabled={tagging || !selectedDatasetForTagging}
                    className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-pink-500 to-pink-600 text-white px-6 py-4 rounded-lg font-semibold hover:from-pink-600 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg text-lg"
                  >
                    <Tag className="w-6 h-6" />
                    {tagging ? 'Tagging in Progress...' : 'Start Auto-Tagging'}
                  </button>

                  {tagging && (
                    <div className="text-sm text-center text-gray-400">
                      ⏳ Processing images... Check logs below for progress
                    </div>
                  )}
                </div>
              </div>

              {/* Log Viewer */}
              <div className="mt-6 border-t border-slate-700 pt-6">
                <button
                  onClick={() => setShowLogs(!showLogs)}
                  className="flex items-center justify-between w-full mb-3 text-white hover:text-cyan-400 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <Terminal className="w-5 h-5 text-cyan-400" />
                    <h3 className="text-lg font-bold">Process Logs</h3>
                    <span className="text-sm text-gray-500">
                      ({logs.length} {logs.length === 1 ? 'entry' : 'entries'})
                    </span>
                  </div>
                  {showLogs ? (
                    <ChevronUp className="w-5 h-5" />
                  ) : (
                    <ChevronDown className="w-5 h-5" />
                  )}
                </button>

                {showLogs && (
                  <div className="bg-slate-950 rounded-lg border border-slate-700 p-4 max-h-80 overflow-y-auto font-mono text-sm">
                    {logs.length === 0 ? (
                      <div className="text-gray-500 text-center py-8">
                        No logs yet. Start a tagging operation to see output here.
                      </div>
                    ) : (
                      <>
                        {logs.map((log, idx) => (
                          <div
                            key={idx}
                            className="text-gray-300 hover:bg-slate-800/50 px-2 py-1 rounded transition-colors"
                          >
                            {log}
                          </div>
                        ))}
                        {tagging && (
                          <div className="text-cyan-400 px-2 py-1 animate-pulse">
                            ▶ Running...
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}

                {!showLogs && (
                  <div className="text-sm text-gray-500 text-center py-2">
                    Click to {showLogs ? 'hide' : 'show'} logs
                  </div>
                )}
              </div>
            </div>
          </GradientCard>

        {/* Info Cards */}
        <div className="mt-8 grid md:grid-cols-3 gap-6">
          <GradientCard variant="dusk" intensity="subtle">
            <div className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="bg-purple-900/50 p-3 rounded-lg">
                  <ImageIcon className="w-6 h-6 text-purple-400" />
                </div>
                <h3 className="font-semibold text-white">Supported Formats</h3>
              </div>
              <p className="text-sm text-gray-300">
                PNG, JPG, JPEG, WebP, BMP
                <br />
                Recommended: 512x512 or 1024x1024
              </p>
            </div>
          </GradientCard>

          <GradientCard variant="cotton-candy" intensity="subtle">
            <div className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="bg-pink-900/50 p-3 rounded-lg">
                  <Tag className="w-6 h-6 text-pink-400" />
                </div>
                <h3 className="font-semibold text-white">Auto-Tagging</h3>
              </div>
              <p className="text-sm text-gray-300">
                Uses WD14 tagger to automatically caption your images with descriptive tags
              </p>
            </div>
          </GradientCard>

          <GradientCard variant="watermelon" intensity="subtle">
            <div className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="bg-green-900/50 p-3 rounded-lg">
                  <FolderOpen className="w-6 h-6 text-green-400" />
                </div>
                <h3 className="font-semibold text-white">Organization</h3>
              </div>
              <p className="text-sm text-gray-300">
                Keep datasets organized in separate folders for easy management and training
              </p>
            </div>
          </GradientCard>
        </div>

        {/* Info Banner */}
        <GradientCard variant="ocean" intensity="medium" className="mt-8">
          <div className="p-6">
            <div className="grid md:grid-cols-3 gap-4 text-sm">
              <div>
                <strong className="block mb-1 text-white">Batch Upload</strong>
                <p className="text-gray-300">
                  Multi-file upload support with progress tracking
                </p>
              </div>
              <div>
                <strong className="block mb-1 text-white">Auto-Tagging</strong>
                <p className="text-gray-300">
                  WD14 tagger integration for automatic captioning
                </p>
              </div>
              <div>
                <strong className="block mb-1 text-white">Tag Management</strong>
                <p className="text-gray-300">
                  Edit, bulk operations, and trigger word injection
                </p>
              </div>
            </div>
          </div>
        </GradientCard>

        {/* Next Steps */}
        <GradientCard variant="cotton-candy" intensity="subtle" className="mt-6">
          <div className="p-6">
            <h3 className="text-xl font-bold text-white mb-4">📋 Next Steps</h3>
            <div className="space-y-3">
              <Link
                href="/dataset/tags"
                className="flex items-center justify-between p-4 bg-slate-800/50 hover:bg-slate-800 rounded-lg border border-slate-600 hover:border-cyan-500 transition-all group"
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-cyan-500/10 rounded-lg group-hover:bg-cyan-500/20 transition-colors">
                    <Tag className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div>
                    <div className="font-semibold text-white">Edit & Manage Tags</div>
                    <div className="text-sm text-gray-400">
                      Review captions, add trigger words, bulk operations
                    </div>
                  </div>
                </div>
                <div className="text-cyan-400 group-hover:translate-x-1 transition-transform">→</div>
              </Link>

              <div className="flex items-center justify-between p-4 bg-slate-800/30 rounded-lg border border-slate-700 opacity-50">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-purple-500/10 rounded-lg">
                    <FolderOpen className="w-5 h-5 text-purple-400" />
                  </div>
                  <div>
                    <div className="font-semibold text-white">Start Training</div>
                    <div className="text-sm text-gray-400">
                      Configure and launch LoRA training (Coming soon)
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </GradientCard>
      </div>
    </div>
  );
}
