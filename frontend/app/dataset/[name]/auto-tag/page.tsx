'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { datasetAPI, captioningAPI, DatasetInfo, BLIPConfig, GITConfig } from '@/lib/api';
import { Home, Database, Tag, Zap, Info, ChevronDown, ChevronUp, Terminal, X, Play, Square, Settings, Sliders, Sparkles, Camera } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

type CaptioningMethod = 'wd14' | 'blip' | 'git';

export default function AutoTagPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(true);

  // Method selection
  const [method, setMethod] = useState<CaptioningMethod>('wd14');

  // Basic settings
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [taggerModel, setTaggerModel] = useState('SmilingWolf/wd-vit-large-tagger-v3');
  const [captionExtension, setCaptionExtension] = useState('.txt');
  const [captionSeparator, setCaptionSeparator] = useState(', ');

  // BLIP settings
  const [blipBeamSearch, setBlipBeamSearch] = useState(false);
  const [blipNumBeams, setBlipNumBeams] = useState(3);
  const [blipTopP, setBlipTopP] = useState(0.9);
  const [blipMaxLength, setBlipMaxLength] = useState(75);
  const [blipMinLength, setBlipMinLength] = useState(5);

  // GIT settings
  const [gitModel, setGitModel] = useState('microsoft/git-large-textcaps');
  const [gitMaxLength, setGitMaxLength] = useState(50);
  const [gitRemoveWords, setGitRemoveWords] = useState(true);

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
        (data) => { if (data.log) setLogs(prev => [...prev, data.log as string]); },
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

  // Start tagging/captioning
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

    const methodLabel = method === 'wd14' ? 'WD14 Tagging' : method === 'blip' ? 'BLIP Captioning' : 'GIT Captioning';
    addLog(`🚀 Starting ${methodLabel}...`);
    addLog(`📁 Dataset: ${selectedDataset}`);

    try {
      let response;

      if (method === 'blip') {
        addLog(`🤖 Model: BLIP (${blipBeamSearch ? 'beam search' : 'nucleus sampling'})`);
        const config: BLIPConfig = {
          dataset_dir: selectedDataset,
          caption_extension: captionExtension,
          batch_size: batchSize,
          max_workers: maxWorkers,
          beam_search: blipBeamSearch,
          num_beams: blipNumBeams,
          top_p: blipTopP,
          max_length: blipMaxLength,
          min_length: blipMinLength,
          recursive,
          debug,
        };
        response = await captioningAPI.startBLIP(config);
      } else if (method === 'git') {
        addLog(`🤖 Model: ${gitModel}`);
        const config: GITConfig = {
          dataset_dir: selectedDataset,
          caption_extension: captionExtension,
          model_id: gitModel,
          batch_size: batchSize,
          max_workers: maxWorkers,
          max_length: gitMaxLength,
          remove_words: gitRemoveWords,
          recursive,
          debug,
        };
        response = await captioningAPI.startGIT(config);
      } else {
        // WD14 Tagging
        addLog(`🤖 Model: ${taggerModel}`);
        addLog(`🎯 Thresholds: Overall=${threshold}, General=${useGeneralThreshold ? generalThreshold : 'default'}, Character=${useCharacterThreshold ? characterThreshold : 'default'}`);
        response = await datasetAPI.tag({
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
      }

      // Handle response (same for all methods)
      if (response.success && response.job_id) {
        setJobId(response.job_id);
        addLog(`✅ Job started! ID: ${response.job_id}`);

        // Connect logs and status polling
        if (method === 'wd14') {
          connectLogs(response.job_id);
        } else {
          // BLIP/GIT use same WebSocket endpoint
          if (wsRef.current) wsRef.current.close();
          wsRef.current = captioningAPI.connectLogs(
            response.job_id,
            (data) => { if (data.log) setLogs(prev => [...prev, data.log as string]); },
            (error) => {
              console.error('WebSocket error:', error);
              addLog('⚠️ Log connection lost - using status polling');
            }
          );
        }

        startStatusPolling(response.job_id);
      } else {
        addLog(`❌ Failed: ${response.message}`);
        setTagging(false);
      }
    } catch (err) {
      addLog(`❌ Error: ${err}`);
      alert(`❌ ${methodLabel} failed: ${err}`);
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
            Auto-Tagging & Captioning
          </h1>
          <p className="text-xl text-muted-foreground mt-4">
            Generate tags or captions for your dataset
          </p>

          {/* Method Selector */}
          <div className="mt-6 grid grid-cols-3 gap-4 max-w-3xl">
            <button
              onClick={() => setMethod('wd14')}
              disabled={tagging}
              className={`p-6 rounded-xl border-2 transition-all ${
                method === 'wd14'
                  ? 'border-pink-500 bg-pink-500/10'
                  : 'border-border hover:border-pink-300'
              } disabled:opacity-50`}
            >
              <Tag className={`w-8 h-8 mx-auto mb-2 ${method === 'wd14' ? 'text-pink-500' : 'text-muted-foreground'}`} />
              <div className="font-bold text-lg">WD14 Tags</div>
              <div className="text-sm text-muted-foreground mt-1">Anime-style booru tags</div>
              <div className="text-xs text-muted-foreground mt-2">girl, blue_eyes, smile</div>
            </button>

            <button
              onClick={() => setMethod('blip')}
              disabled={tagging}
              className={`p-6 rounded-xl border-2 transition-all ${
                method === 'blip'
                  ? 'border-cyan-500 bg-cyan-500/10'
                  : 'border-border hover:border-cyan-300'
              } disabled:opacity-50`}
            >
              <Sparkles className={`w-8 h-8 mx-auto mb-2 ${method === 'blip' ? 'text-cyan-500' : 'text-muted-foreground'}`} />
              <div className="font-bold text-lg">BLIP</div>
              <div className="text-sm text-muted-foreground mt-1">Natural language</div>
              <div className="text-xs text-muted-foreground mt-2">a girl with blue eyes smiling</div>
            </button>

            <button
              onClick={() => setMethod('git')}
              disabled={tagging}
              className={`p-6 rounded-xl border-2 transition-all ${
                method === 'git'
                  ? 'border-purple-500 bg-purple-500/10'
                  : 'border-border hover:border-purple-300'
              } disabled:opacity-50`}
            >
              <Camera className={`w-8 h-8 mx-auto mb-2 ${method === 'git' ? 'text-purple-500' : 'text-muted-foreground'}`} />
              <div className="font-bold text-lg">GIT</div>
              <div className="text-sm text-muted-foreground mt-1">Photo captions</div>
              <div className="text-xs text-muted-foreground mt-2">a portrait of a girl with blue eyes</div>
            </button>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left Column: Basic Settings */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5" />
                  Basic Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">

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
              </CardContent>
            </Card>

            {/* Dataset Info */}
            {selectedDatasetInfo && (
              <Card>
                <CardHeader>
                  <CardTitle>Selected Dataset</CardTitle>
                </CardHeader>
                <CardContent>
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
                </CardContent>
              </Card>
            )}
          </div>

          {/* Middle Column: Method-Specific Settings */}
          <div className="space-y-6">
            {method === 'wd14' && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sliders className="w-5 h-5" />
                    Thresholds
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">

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
              </CardContent>
            </Card>
            )}

            {method === 'blip' && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="w-5 h-5" />
                    BLIP Settings
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">

                  {/* Sampling Method */}
                  <div>
                    <label className="block text-sm font-medium mb-2">Sampling Method</label>
                    <div className="flex gap-4">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          checked={!blipBeamSearch}
                          onChange={() => setBlipBeamSearch(false)}
                          disabled={tagging}
                          className="accent-cyan-500"
                        />
                        <span className="text-sm">Nucleus (faster)</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          checked={blipBeamSearch}
                          onChange={() => setBlipBeamSearch(true)}
                          disabled={tagging}
                          className="accent-cyan-500"
                        />
                        <span className="text-sm">Beam Search (better)</span>
                      </label>
                    </div>
                  </div>

                  {/* Beam Settings */}
                  {blipBeamSearch && (
                    <div>
                      <label className="block text-sm font-medium mb-2">Number of Beams: {blipNumBeams}</label>
                      <input
                        type="range"
                        min="1"
                        max="10"
                        value={blipNumBeams}
                        onChange={(e) => setBlipNumBeams(parseInt(e.target.value))}
                        disabled={tagging}
                        className="w-full accent-cyan-500"
                      />
                      <p className="text-xs text-muted-foreground mt-2">More beams = better quality but slower</p>
                    </div>
                  )}

                  {/* Top-P */}
                  {!blipBeamSearch && (
                    <div>
                      <label className="block text-sm font-medium mb-2">Top-P (Nucleus): {blipTopP.toFixed(2)}</label>
                      <input
                        type="range"
                        min="0.5"
                        max="1.0"
                        step="0.05"
                        value={blipTopP}
                        onChange={(e) => setBlipTopP(parseFloat(e.target.value))}
                        disabled={tagging}
                        className="w-full accent-cyan-500"
                      />
                    </div>
                  )}

                  {/* Length */}
                  <div>
                    <label className="block text-sm font-medium mb-2">Max Length: {blipMaxLength}</label>
                    <input
                      type="range"
                      min="20"
                      max="150"
                      step="5"
                      value={blipMaxLength}
                      onChange={(e) => setBlipMaxLength(parseInt(e.target.value))}
                      disabled={tagging}
                      className="w-full accent-cyan-500"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2">Min Length: {blipMinLength}</label>
                    <input
                      type="range"
                      min="1"
                      max="20"
                      value={blipMinLength}
                      onChange={(e) => setBlipMinLength(parseInt(e.target.value))}
                      disabled={tagging}
                      className="w-full accent-cyan-500"
                    />
                  </div>
                </CardContent>
              </Card>
            )}

            {method === 'git' && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Camera className="w-5 h-5" />
                    GIT Settings
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">

                  {/* Model */}
                  <div>
                    <label className="block text-sm font-medium mb-2">Model</label>
                    <select
                      value={gitModel}
                      onChange={(e) => setGitModel(e.target.value)}
                      disabled={tagging}
                      className="w-full px-4 py-2 bg-input border border-border rounded-lg focus:ring-2 focus:ring-purple-500 disabled:opacity-50"
                    >
                      <option value="microsoft/git-large-textcaps">GIT Large TextCaps (best)</option>
                      <option value="microsoft/git-large">GIT Large</option>
                      <option value="microsoft/git-base">GIT Base (faster)</option>
                    </select>
                  </div>

                  {/* Max Length */}
                  <div>
                    <label className="block text-sm font-medium mb-2">Max Caption Length: {gitMaxLength}</label>
                    <input
                      type="range"
                      min="20"
                      max="100"
                      step="5"
                      value={gitMaxLength}
                      onChange={(e) => setGitMaxLength(parseInt(e.target.value))}
                      disabled={tagging}
                      className="w-full accent-purple-500"
                    />
                  </div>

                  {/* Remove Words */}
                  <div className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      id="git-remove-words"
                      checked={gitRemoveWords}
                      onChange={(e) => setGitRemoveWords(e.target.checked)}
                      disabled={tagging}
                      className="w-4 h-4 accent-purple-500"
                    />
                    <label htmlFor="git-remove-words" className="text-sm cursor-pointer">
                      Remove "with the words xxx" artifacts
                    </label>
                  </div>

                  <div className="pt-4 border-t border-border">
                    <p className="text-xs text-muted-foreground">
                      💡 GIT is optimized for photo-realistic images and general scenes
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}

            {method === 'wd14' && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Tag className="w-5 h-5" />
                  Tag Processing
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">

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
              </CardContent>
            </Card>
            )}
          </div>

          {/* Right Column: Performance & Actions */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="w-5 h-5" />
                  Performance
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">

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
              </CardContent>
            </Card>

            {/* Action Buttons */}
            <Card>
              <CardContent className="pt-6 space-y-4">
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
              </CardContent>
            </Card>

            {/* How It Works */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Info className="w-5 h-5" />
                  How It Works
                </CardTitle>
              </CardHeader>
              <CardContent>
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
              </CardContent>
            </Card>
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
