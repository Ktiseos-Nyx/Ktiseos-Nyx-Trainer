'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { datasetAPI, DatasetInfo } from '@/lib/api';
import { Home, Database, Tag, Zap, Info, ChevronDown, ChevronUp, Terminal, X, Play, Square, Settings, Sliders } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { GradientCard } from '@/components/effects';

export default function AutoTagPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(true);

  // Basic settings
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [taggerModel, setTaggerModel] = useState('SmilingWolf/wd-vit-large-tagger-v3');
  const [captionExtension, setCaptionExtension] = useState('.txt');
  const [captionSeparator, setCaptionSeparator] = useState(', ');

  // Thresholds (3 separate!)
  const [threshold, setThreshold] = useState(0.35);
  const [useGeneralThreshold, setUseGeneralThreshold] = useState(false);
  const [generalThreshold, setGeneralThreshold] = useState(0.35);
  const [useCharacterThreshold, setUseCharacterThreshold] = useState(false);
  const [characterThreshold, setCharacterThreshold] = useState(0.35);

  // Tag filtering
  const [undesiredTags, setUndesiredTags] = useState('');
  const [tagReplacement, setTagReplacement] = useState('');

  // Tag ordering
  const [alwaysFirstTags, setAlwaysFirstTags] = useState('');
  const [characterTagsFirst, setCharacterTagsFirst] = useState(false);

  // Rating tags
  const [ratingTags, setRatingTags] = useState<'none' | 'first' | 'last'>('none');

  // Tag processing
  const [removeUnderscore, setRemoveUnderscore] = useState(true);
  const [characterTagExpand, setCharacterTagExpand] = useState(false);

  // File handling
  const [appendTags, setAppendTags] = useState(false);
  const [recursive, setRecursive] = useState(false);

  // Performance
  const [batchSize, setBatchSize] = useState(8);
  const [maxWorkers, setMaxWorkers] = useState(2);
  const [useOnnx, setUseOnnx] = useState(true);
  const [forceDownload, setForceDownload] = useState(false);

  // Debug
  const [frequencyTags, setFrequencyTags] = useState(false);
  const [debug, setDebug] = useState(false);

  // UI state
  const [tagging, setTagging] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Job tracking
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [currentImage, setCurrentImage] = useState<string | null>(null);
  const [totalImages, setTotalImages] = useState<number | null>(null);

  // Log viewer
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const statusIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Auto-enable ONNX for v3 models (smart UX!)
  useEffect(() => {
    if (taggerModel.includes('-v3')) {
      setUseOnnx(true);
    }
  }, [taggerModel]);

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

  // Cleanup
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);
    };
  }, []);

  const addLog = (msg: string) => {
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
  };

  // Poll status
  const startStatusPolling = (jobId: string) => {
    if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);
    statusIntervalRef.current = setInterval(async () => {
      try {
        const status = await datasetAPI.getTaggingStatus(jobId);
        setProgress(status.progress);
        setCurrentImage(status.current_image);
        setTotalImages(status.total_images);
        if (status.status === 'completed') {
          addLog('✅ Tagging completed successfully!');
          setTagging(false);
          if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);
          setTimeout(loadDatasets, 1000);
        } else if (status.status === 'failed') {
          addLog(`❌ Tagging failed: ${status.error}`);
          setTagging(false);
          if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);
        }
      } catch (err) {
        console.error('Status poll error:', err);
      }
    }, 1000);
  };

  // WebSocket logs
  const connectLogs = (jobId: string) => {
    if (wsRef.current) wsRef.current.close();
    try {
      wsRef.current = datasetAPI.connectTaggingLogs(
        jobId,
        (data) => { if (data.log) setLogs(prev => [...prev, data.log]); },
        (error) => {
          console.error('WebSocket error:', error);
          addLog('⚠️ Log connection lost - using status polling');
        }
      );
    } catch (err) {
      console.error('WebSocket connect error:', err);
      addLog('⚠️ Real-time logs unavailable');
    }
  };

  // Start tagging
  const handleStartTagging = async () => {
    if (!selectedDataset) {
      alert('Please select a dataset!');
      return;
    }

    setTagging(true);
    setShowLogs(true);
    setLogs([]);
    setProgress(0);
    setCurrentImage(null);
    setTotalImages(null);

    addLog('🚀 Starting auto-tagging...');
    addLog(`📁 Dataset: ${selectedDataset}`);
    addLog(`🤖 Model: ${taggerModel}`);
    addLog(`🎯 Thresholds: Overall=${threshold}, General=${useGeneralThreshold ? generalThreshold : 'default'}, Character=${useCharacterThreshold ? characterThreshold : 'default'}`);

    try {
      const response = await datasetAPI.tag({
        datasetDir: selectedDataset,
        model: taggerModel,
        forceDownload,
        threshold,
        generalThreshold: useGeneralThreshold ? generalThreshold : null,
        characterThreshold: useCharacterThreshold ? characterThreshold : null,
        captionExtension,
        captionSeparator,
        undesiredTags,
        tagReplacement: tagReplacement || null,
        alwaysFirstTags: alwaysFirstTags || null,
        characterTagsFirst,
        useRatingTags: ratingTags !== 'none',
        useRatingTagsAsLastTag: ratingTags === 'last',
        removeUnderscore,
        characterTagExpand,
        appendTags,
        recursive,
        batchSize,
        maxWorkers,
        useOnnx,
        frequencyTags,
        debug,
      });

      if (response.success && response.job_id) {
        setJobId(response.job_id);
        addLog(`✅ Job started! ID: ${response.job_id}`);
        connectLogs(response.job_id);
        startStatusPolling(response.job_id);
      } else {
        addLog(`❌ Failed: ${response.message}`);
        setTagging(false);
      }
    } catch (err) {
      addLog(`❌ Error: ${err}`);
      alert(`❌ Tagging failed: ${err}`);
      setTagging(false);
    }
  };

  // Stop tagging
  const handleStopTagging = async () => {
    if (!jobId) return;
    try {
      addLog('🛑 Stopping...');
      await datasetAPI.stopTagging(jobId);
      addLog('✅ Stopped');
      setTagging(false);
      if (wsRef.current) wsRef.current.close();
      if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);
    } catch (err) {
      addLog(`❌ Stop failed: ${err}`);
    }
  };

  const selectedDatasetInfo = datasets.find(d => d.path === selectedDataset);

  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Dataset', href: '/dataset', icon: <Database className="w-4 h-4" /> },
            { label: 'Auto-Tagging', icon: <Tag className="w-4 h-4" /> },
          ]}
        />

        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-pink-400 via-rose-400 to-pink-400 bg-clip-text text-transparent">
            WD14 Auto-Tagging
          </h1>
          <p className="text-xl text-muted-foreground mt-4">
            Automatic caption generation with comprehensive control over tag processing
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left Column: Basic Settings */}
          <div className="space-y-6">
            <GradientCard variant="watermelon" intensity="subtle">
              <div className="p-6 space-y-6">
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <Zap className="w-6 h-6 text-pink-600 dark:text-pink-400" />
                  Basic Settings
                </h2>

                {/* Dataset */}
                <div>
                  <label className="block text-sm font-medium mb-2">Dataset</label>
                  {loading ? (
                    <div className="text-sm text-muted-foreground">Loading...</div>
                  ) : datasets.length === 0 ? (
                    <div className="text-sm text-muted-foreground">
                      No datasets. <Link href="/dataset" className="text-pink-600 hover:underline">Upload images</Link>
                    </div>
                  ) : (
                    <select
                      value={selectedDataset}
                      onChange={(e) => setSelectedDataset(e.target.value)}
                      disabled={tagging}
                      className="w-full px-4 py-2 bg-input border border-border rounded-lg focus:ring-2 focus:ring-pink-500 disabled:opacity-50"
                    >
                      {datasets.map(ds => (
                        <option key={ds.path} value={ds.path}>{ds.name} ({ds.image_count} images)</option>
                      ))}
                    </select>
                  )}
                </div>

                {/* Model */}
                <div>
                  <label className="block text-sm font-medium mb-2">Tagger Model</label>
                  <select
                    value={taggerModel}
                    onChange={(e) => setTaggerModel(e.target.value)}
                    disabled={tagging}
                    className="w-full px-4 py-2 bg-input border border-border rounded-lg focus:ring-2 focus:ring-pink-500 disabled:opacity-50"
                  >
                    <option value="SmilingWolf/wd-vit-large-tagger-v3">WD14 ViT Large v3 ⭐</option>
                    <option value="SmilingWolf/wd-vit-tagger-v3">WD14 ViT v3</option>
                    <option value="SmilingWolf/wd-v1-4-swinv2-tagger-v2">WD14 SwinV2 v2</option>
                    <option value="SmilingWolf/wd-v1-4-convnext-tagger-v2">WD14 ConvNext v2</option>
                    <option value="SmilingWolf/wd-v1-4-vit-tagger-v2">WD14 ViT v2</option>
                  </select>
                  <p className="text-xs text-muted-foreground mt-2">
                    v3 models auto-enable ONNX for best performance
                  </p>
                </div>

                {/* Caption Extension */}
                <div>
                  <label className="block text-sm font-medium mb-2">Caption Extension</label>
                  <select
                    value={captionExtension}
                    onChange={(e) => setCaptionExtension(e.target.value)}
                    disabled={tagging}
                    className="w-full px-4 py-2 bg-input border border-border rounded-lg focus:ring-2 focus:ring-pink-500 disabled:opacity-50"
                  >
                    <option value=".txt">.txt (Kohya standard)</option>
                    <option value=".caption">.caption</option>
                    <option value=".cap">.cap</option>
                  </select>
                </div>

                {/* Caption Separator */}
                <div>
                  <label className="block text-sm font-medium mb-2">Tag Separator</label>
                  <input
                    type="text"
                    value={captionSeparator}
                    onChange={(e) => setCaptionSeparator(e.target.value)}
                    disabled={tagging}
                    placeholder=", "
                    className="w-full px-4 py-2 bg-input border border-border rounded-lg focus:ring-2 focus:ring-pink-500 disabled:opacity-50"
                  />
                </div>
              </div>
            </GradientCard>

            {/* Dataset Info */}
            {selectedDatasetInfo && (
              <GradientCard variant="shadow" intensity="subtle">
                <div className="p-6">
                  <h3 className="text-xl font-bold mb-4">Selected Dataset</h3>
                  <div className="space-y-2 text-sm">
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
                          <span className="text-green-400">Has tags (will {appendTags ? 'append' : 'overwrite'})</span>
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

          {/* Middle Column: Thresholds & Tag Processing */}
          <div className="space-y-6">
            <GradientCard variant="ocean" intensity="subtle">
              <div className="p-6 space-y-6">
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <Sliders className="w-6 h-6 text-cyan-600 dark:text-cyan-400" />
                  Thresholds
                </h2>

                {/* Overall Threshold */}
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Overall Threshold: {threshold.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    disabled={tagging}
                    className="w-full accent-cyan-500 disabled:opacity-50"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>More tags</span>
                    <span>Fewer tags</span>
                  </div>
                </div>

                {/* General Threshold */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <input
                      type="checkbox"
                      id="use-general"
                      checked={useGeneralThreshold}
                      onChange={(e) => setUseGeneralThreshold(e.target.checked)}
                      disabled={tagging}
                      className="w-4 h-4 accent-cyan-500"
                    />
                    <label htmlFor="use-general" className="text-sm font-medium cursor-pointer">
                      Custom General Threshold: {generalThreshold.toFixed(2)}
                    </label>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={generalThreshold}
                    onChange={(e) => setGeneralThreshold(parseFloat(e.target.value))}
                    disabled={tagging || !useGeneralThreshold}
                    className="w-full accent-cyan-500 disabled:opacity-50"
                  />
                </div>

                {/* Character Threshold */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <input
                      type="checkbox"
                      id="use-character"
                      checked={useCharacterThreshold}
                      onChange={(e) => setUseCharacterThreshold(e.target.checked)}
                      disabled={tagging}
                      className="w-4 h-4 accent-cyan-500"
                    />
                    <label htmlFor="use-character" className="text-sm font-medium cursor-pointer">
                      Custom Character Threshold: {characterThreshold.toFixed(2)}
                    </label>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={characterThreshold}
                    onChange={(e) => setCharacterThreshold(parseFloat(e.target.value))}
                    disabled={tagging || !useCharacterThreshold}
                    className="w-full accent-cyan-500 disabled:opacity-50"
                  />
                </div>

                <div className="pt-4 border-t border-border">
                  <p className="text-xs text-muted-foreground">
                    💡 Separate thresholds for general/character tags allow fine-tuned control
                  </p>
                </div>
              </div>
            </GradientCard>

            <GradientCard variant="watermelon" intensity="subtle">
              <div className="p-6 space-y-6">
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <Tag className="w-6 h-6 text-orange-600 dark:text-orange-400" />
                  Tag Processing
                </h2>

                {/* Prefix Tags */}
                <div>
                  <label className="block text-sm font-medium mb-2">Prefix Tags (Optional)</label>
                  <input
                    type="text"
                    value={alwaysFirstTags}
                    onChange={(e) => setAlwaysFirstTags(e.target.value)}
                    disabled={tagging}
                    placeholder="e.g., 1girl, solo"
                    className="w-full px-4 py-2 bg-input border border-border rounded-lg focus:ring-2 focus:ring-orange-500 disabled:opacity-50"
                  />
                  <p className="text-xs text-muted-foreground mt-2">
                    Tags to always put first
                  </p>
                </div>

                {/* Blacklist */}
                <div>
                  <label className="block text-sm font-medium mb-2">Undesired Tags (Optional)</label>
                  <input
                    type="text"
                    value={undesiredTags}
                    onChange={(e) => setUndesiredTags(e.target.value)}
                    disabled={tagging}
                    placeholder="e.g., watermark, logo, text"
                    className="w-full px-4 py-2 bg-input border border-border rounded-lg focus:ring-2 focus:ring-orange-500 disabled:opacity-50"
                  />
                  <p className="text-xs text-muted-foreground mt-2">
                    Comma-separated tags to exclude
                  </p>
                </div>

                {/* Tag Replacement */}
                <div>
                  <label className="block text-sm font-medium mb-2">Tag Replacement (Optional)</label>
                  <input
                    type="text"
                    value={tagReplacement}
                    onChange={(e) => setTagReplacement(e.target.value)}
                    disabled={tagging}
                    placeholder="old1,new1;old2,new2"
                    className="w-full px-4 py-2 bg-input border border-border rounded-lg focus:ring-2 focus:ring-orange-500 disabled:opacity-50"
                  />
                  <p className="text-xs text-muted-foreground mt-2">
                    Format: source,target;source,target
                  </p>
                </div>

                {/* Rating Tags */}
                <div>
                  <label className="block text-sm font-medium mb-2">Rating Tags</label>
                  <div className="flex gap-4">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name="rating"
                        value="none"
                        checked={ratingTags === 'none'}
                        onChange={() => setRatingTags('none')}
                        disabled={tagging}
                        className="accent-orange-500"
                      />
                      <span className="text-sm">None</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name="rating"
                        value="first"
                        checked={ratingTags === 'first'}
                        onChange={() => setRatingTags('first')}
                        disabled={tagging}
                        className="accent-orange-500"
                      />
                      <span className="text-sm">First</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name="rating"
                        value="last"
                        checked={ratingTags === 'last'}
                        onChange={() => setRatingTags('last')}
                        disabled={tagging}
                        className="accent-orange-500"
                      />
                      <span className="text-sm">Last</span>
                    </label>
                  </div>
                </div>

                {/* Flags */}
                <div className="space-y-2">
                  <div className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      id="char-first"
                      checked={characterTagsFirst}
                      onChange={(e) => setCharacterTagsFirst(e.target.checked)}
                      disabled={tagging}
                      className="w-4 h-4 accent-orange-500"
                    />
                    <label htmlFor="char-first" className="text-sm cursor-pointer">
                      Character Tags First
                    </label>
                  </div>
                  <div className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      id="rm-underscore"
                      checked={removeUnderscore}
                      onChange={(e) => setRemoveUnderscore(e.target.checked)}
                      disabled={tagging}
                      className="w-4 h-4 accent-orange-500"
                    />
                    <label htmlFor="rm-underscore" className="text-sm cursor-pointer">
                      Remove Underscores
                    </label>
                  </div>
                  <div className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      id="char-expand"
                      checked={characterTagExpand}
                      onChange={(e) => setCharacterTagExpand(e.target.checked)}
                      disabled={tagging}
                      className="w-4 h-4 accent-orange-500"
                    />
                    <label htmlFor="char-expand" className="text-sm cursor-pointer">
                      Expand Character Tags (name_(series) → name, series)
                    </label>
                  </div>
                </div>
              </div>
            </GradientCard>
          </div>

          {/* Right Column: Performance & Actions */}
          <div className="space-y-6">
            <GradientCard variant="cotton-candy" intensity="subtle">
              <div className="p-6 space-y-6">
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <Settings className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                  Performance
                </h2>

                {/* Batch Size */}
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Batch Size: {batchSize}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="16"
                    step="1"
                    value={batchSize}
                    onChange={(e) => setBatchSize(parseInt(e.target.value))}
                    disabled={tagging}
                    className="w-full accent-purple-500 disabled:opacity-50"
                  />
                  <p className="text-xs text-muted-foreground mt-2">
                    Higher = faster (more GPU memory)
                  </p>
                </div>

                {/* Max Workers */}
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Data Loader Workers: {maxWorkers}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="8"
                    step="1"
                    value={maxWorkers}
                    onChange={(e) => setMaxWorkers(parseInt(e.target.value))}
                    disabled={tagging}
                    className="w-full accent-purple-500 disabled:opacity-50"
                  />
                </div>

                {/* Flags */}
                <div className="space-y-2">
                  <div className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      id="onnx"
                      checked={useOnnx}
                      onChange={(e) => setUseOnnx(e.target.checked)}
                      disabled={tagging}
                      className="w-4 h-4 accent-purple-500"
                    />
                    <label htmlFor="onnx" className="text-sm cursor-pointer">
                      Use ONNX Runtime (faster)
                    </label>
                  </div>
                  <div className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      id="append"
                      checked={appendTags}
                      onChange={(e) => setAppendTags(e.target.checked)}
                      disabled={tagging}
                      className="w-4 h-4 accent-purple-500"
                    />
                    <label htmlFor="append" className="text-sm cursor-pointer">
                      Append Tags (don't overwrite)
                    </label>
                  </div>
                  <div className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      id="recursive"
                      checked={recursive}
                      onChange={(e) => setRecursive(e.target.checked)}
                      disabled={tagging}
                      className="w-4 h-4 accent-purple-500"
                    />
                    <label htmlFor="recursive" className="text-sm cursor-pointer">
                      Recursive (process subfolders)
                    </label>
                  </div>
                  <div className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      id="freq"
                      checked={frequencyTags}
                      onChange={(e) => setFrequencyTags(e.target.checked)}
                      disabled={tagging}
                      className="w-4 h-4 accent-purple-500"
                    />
                    <label htmlFor="freq" className="text-sm cursor-pointer">
                      Show Tag Frequency Report
                    </label>
                  </div>
                </div>

                {/* Advanced Options Toggle */}
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="w-full px-4 py-2 bg-accent hover:bg-accent/80 rounded-lg flex items-center justify-between"
                >
                  <span className="text-sm font-medium">Advanced Options</span>
                  {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>

                {showAdvanced && (
                  <div className="space-y-2 pt-2 border-t border-border">
                    <div className="flex items-center gap-3">
                      <input
                        type="checkbox"
                        id="force-dl"
                        checked={forceDownload}
                        onChange={(e) => setForceDownload(e.target.checked)}
                        disabled={tagging}
                        className="w-4 h-4 accent-purple-500"
                      />
                      <label htmlFor="force-dl" className="text-sm cursor-pointer">
                        Force Download Model
                      </label>
                    </div>
                    <div className="flex items-center gap-3">
                      <input
                        type="checkbox"
                        id="debug"
                        checked={debug}
                        onChange={(e) => setDebug(e.target.checked)}
                        disabled={tagging}
                        className="w-4 h-4 accent-purple-500"
                      />
                      <label htmlFor="debug" className="text-sm cursor-pointer">
                        Debug Mode
                      </label>
                    </div>
                  </div>
                )}
              </div>
            </GradientCard>

            {/* Action Buttons */}
            <GradientCard variant="watermelon" intensity="subtle">
              <div className="p-6 space-y-4">
                <div className="flex gap-3">
                  <button
                    onClick={handleStartTagging}
                    disabled={tagging || !selectedDataset || datasets.length === 0}
                    className="flex-1 px-6 py-3 bg-gradient-to-r from-pink-500 to-rose-500 hover:from-pink-600 hover:to-rose-600 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed text-white rounded-lg font-semibold shadow-lg flex items-center justify-center gap-2"
                  >
                    <Play className="w-5 h-5" />
                    {tagging ? 'Tagging...' : 'Start'}
                  </button>
                  {tagging && (
                    <button
                      onClick={handleStopTagging}
                      className="px-6 py-3 bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white rounded-lg font-semibold shadow-lg flex items-center justify-center gap-2"
                    >
                      <Square className="w-5 h-5" />
                      Stop
                    </button>
                  )}
                </div>

                {/* Progress */}
                {tagging && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium">Progress: {progress}%</span>
                      {totalImages && <span className="text-muted-foreground">{totalImages} images</span>}
                    </div>
                    <div className="w-full bg-muted rounded-full h-2">
                      <div
                        className="h-full bg-gradient-to-r from-pink-500 to-rose-500 rounded-full transition-all"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                    {currentImage && (
                      <p className="text-xs text-muted-foreground truncate">
                        {currentImage}
                      </p>
                    )}
                  </div>
                )}
              </div>
            </GradientCard>

            {/* How It Works */}
            <GradientCard variant="ocean" intensity="subtle">
              <div className="p-6">
                <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                  <Info className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
                  How It Works
                </h3>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <span className="text-cyan-400">1.</span>
                    <span>WD14 analyzes images with deep learning</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-cyan-400">2.</span>
                    <span>Generates tags based on confidence thresholds</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-cyan-400">3.</span>
                    <span>Processes tags (ordering, filtering, replacement)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-cyan-400">4.</span>
                    <span>Saves as .txt files (Kohya format)</span>
                  </li>
                </ul>
              </div>
            </GradientCard>
          </div>
        </div>

        {/* Logs */}
        {showLogs && (
          <div className="mt-6">
            <div className="bg-card border border-border rounded-lg overflow-hidden">
              <button
                onClick={() => setShowLogs(!showLogs)}
                className="w-full px-6 py-4 flex items-center justify-between bg-accent/50 hover:bg-accent"
              >
                <div className="flex items-center gap-2 font-semibold">
                  <Terminal className="w-5 h-5" />
                  Process Logs {jobId && <span className="text-xs text-muted-foreground">({jobId})</span>}
                </div>
                <div className="flex items-center gap-2">
                  {logs.length > 0 && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setLogs([]);
                      }}
                      className="p-1 hover:bg-accent-foreground/10 rounded"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                  {showLogs ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                </div>
              </button>
              {showLogs && (
                <div className="p-4 bg-black/50 font-mono text-sm text-green-400 max-h-96 overflow-y-auto">
                  {logs.length === 0 ? (
                    <div className="text-muted-foreground">No logs yet...</div>
                  ) : (
                    logs.map((log, idx) => (
                      <div key={idx} className="py-1">{log}</div>
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
