'use client';

import { useState, useEffect } from 'react';
import { utilitiesAPI, LoRAFile } from '@/lib/api';
import { Wrench, Upload, FolderOpen, CheckCircle, XCircle, Loader2, Minimize2, Home } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';

export default function UtilitiesPage() {
  const [activeTab, setActiveTab] = useState<'merge' | 'resize' | 'checkpoint'>('merge');

  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Utilities', icon: <Wrench className="w-4 h-4" /> },
          ]}
        />

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-orange-400 via-red-400 to-orange-400 bg-clip-text text-transparent">
            Model Utilities
          </h1>
          <p className="text-xl text-muted-foreground">
            Merge, optimize, and publish your trained models
          </p>
        </div>

        {/* Tabs */}
        <div className="mb-6">
          <div className="border-b border-border">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setActiveTab('merge')}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'merge'
                    ? 'border-cyan-400 text-cyan-400'
                    : 'border-transparent text-muted-foreground hover:text-foreground hover:border-input'
                }`}
              >
                <FolderOpen className="w-5 h-5 inline mr-2" />
                Merge LoRAs
              </button>
              <button
                onClick={() => setActiveTab('checkpoint')}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'checkpoint'
                    ? 'border-cyan-400 text-cyan-400'
                    : 'border-transparent text-muted-foreground hover:text-foreground hover:border-input'
                }`}
              >
                <FolderOpen className="w-5 h-5 inline mr-2" />
                Merge Checkpoints
              </button>
              <button
                onClick={() => setActiveTab('resize')}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'resize'
                    ? 'border-cyan-400 text-cyan-400'
                    : 'border-transparent text-muted-foreground hover:text-foreground hover:border-input'
                }`}
              >
                <Minimize2 className="w-5 h-5 inline mr-2" />
                Resize LoRA
              </button>
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'merge' && <MergeLoRATab />}
        {activeTab === 'checkpoint' && <MergeCheckpointTab />}
        {activeTab === 'resize' && <ResizeLoRATab />}
      </div>
    </div>
  );
}

// ========== Merge LoRAs Tab ==========

function MergeLoRATab() {
  const [availableFiles, setAvailableFiles] = useState<LoRAFile[]>([]);
  const [selectedLoras, setSelectedLoras] = useState<Array<{ path: string; name: string; ratio: number }>>([]);
  const [outputPath, setOutputPath] = useState('');
  const [modelType, setModelType] = useState<'sd' | 'sdxl' | 'flux' | 'svd'>('sdxl');
  const [merging, setMerging] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Load available files
  useEffect(() => {
    const loadFiles = async () => {
      try {
        const dirsResponse = await utilitiesAPI.getDirectories();
        const loraDir = dirsResponse.output || 'output';

        const filesResponse = await utilitiesAPI.listLoraFiles(loraDir, 'safetensors', 'date');
        if (filesResponse.success) {
          setAvailableFiles(filesResponse.files);
        }
      } catch (err) {
        console.error('Failed to load LoRA files:', err);
      }
    };
    loadFiles();
  }, []);

  // Auto-generate output path when selection changes
  useEffect(() => {
    if (selectedLoras.length >= 2) {
      const timestamp = new Date().toISOString().split('T')[0];
      setOutputPath(`output/merged_${modelType}_${timestamp}.safetensors`);
    }
  }, [selectedLoras.length, modelType]);

  const addLoraToMerge = (file: LoRAFile) => {
    if (!selectedLoras.find(l => l.path === file.path)) {
      setSelectedLoras([...selectedLoras, { path: file.path, name: file.name, ratio: 1.0 }]);
    }
  };

  const removeLoraFromMerge = (path: string) => {
    setSelectedLoras(selectedLoras.filter(l => l.path !== path));
  };

  const updateRatio = (path: string, ratio: number) => {
    setSelectedLoras(
      selectedLoras.map(l =>
        l.path === path ? { ...l, ratio: Math.max(0, Math.min(2, ratio)) } : l
      )
    );
  };

  const handleMerge = async () => {
    if (selectedLoras.length < 2) {
      setError('Please select at least 2 LoRAs to merge');
      return;
    }

    if (!outputPath) {
      setError('Please provide an output path');
      return;
    }

    try {
      setMerging(true);
      setError(null);
      setResult(null);

      const response = await utilitiesAPI.mergeLora(
        selectedLoras.map(l => ({ path: l.path, ratio: l.ratio })),
        outputPath,
        modelType
      );

      if (response.success) {
        setResult(response);
        setSelectedLoras([]);
      } else {
        setError(response.message || 'Merge failed');
      }
    } catch (err: any) {
      setError(err.message || 'Merge failed');
    } finally {
      setMerging(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Model Type Selector */}
      <div className="bg-card backdrop-blur-sm border border-border rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold text-foreground mb-4">Model Architecture</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {(['sd', 'sdxl', 'flux', 'svd'] as const).map((type) => (
            <button
              key={type}
              onClick={() => setModelType(type)}
              className={`p-4 rounded-lg border-2 transition-all ${
                modelType === type
                  ? 'border-cyan-500 bg-cyan-500/10 text-cyan-400'
                  : 'border-input bg-card/50 text-muted-foreground hover:border-slate-500'
              }`}
            >
              <div className="text-center font-semibold">{type.toUpperCase()}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Available LoRAs */}
      <div className="bg-card backdrop-blur-sm border border-border rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold text-foreground mb-4">Available LoRAs</h2>
        {availableFiles.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            No LoRA files found in output directory
          </div>
        ) : (
          <div className="grid md:grid-cols-2 gap-2 max-h-96 overflow-y-auto">
            {availableFiles.map((file) => (
              <button
                key={file.path}
                onClick={() => addLoraToMerge(file)}
                disabled={selectedLoras.some(l => l.path === file.path)}
                className="flex items-center justify-between p-3 border border-border rounded-lg hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed text-left"
              >
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-foreground truncate">{file.name}</div>
                  <div className="text-sm text-muted-foreground">{file.size_formatted}</div>
                </div>
                {selectedLoras.some(l => l.path === file.path) && (
                  <CheckCircle className="w-5 h-5 text-green-500 ml-2" />
                )}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Selected LoRAs for Merge */}
      {selectedLoras.length > 0 && (
        <div className="bg-card backdrop-blur-sm border border-border rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold text-foreground mb-4">
            Selected LoRAs ({selectedLoras.length})
          </h2>
          <div className="space-y-3">
            {selectedLoras.map((lora) => (
              <div key={lora.path} className="flex items-center gap-4 p-3 border border-border rounded-lg">
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-foreground truncate">{lora.name}</div>
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-sm text-muted-foreground whitespace-nowrap">Ratio:</label>
                  <input
                    type="number"
                    value={lora.ratio}
                    onChange={(e) => updateRatio(lora.path, parseFloat(e.target.value) || 0)}
                    step="0.1"
                    min="0"
                    max="2"
                    className="w-20 px-2 py-1 bg-input border border-input text-foreground rounded text-center"
                  />
                </div>
                <button
                  onClick={() => removeLoraFromMerge(lora.path)}
                  className="p-2 hover:bg-destructive/10 rounded text-destructive"
                  title="Remove"
                >
                  <XCircle className="w-5 h-5" />
                </button>
              </div>
            ))}
          </div>

          {/* Output Path */}
          <div className="mt-4">
            <label className="block text-sm font-medium text-foreground mb-2">Output Path</label>
            <input
              type="text"
              value={outputPath}
              onChange={(e) => setOutputPath(e.target.value)}
              className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg"
              placeholder="output/merged.safetensors"
            />
          </div>
        </div>
      )}

      {/* Merge Button */}
      <button
        onClick={handleMerge}
        disabled={merging || selectedLoras.length < 2 || !outputPath}
        className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-purple-500 to-indigo-600 text-white px-6 py-4 rounded-lg font-semibold hover:from-purple-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
      >
        {merging ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Merging...
          </>
        ) : (
          <>
            <FolderOpen className="w-5 h-5" />
            Merge {selectedLoras.length} LoRAs
          </>
        )}
      </button>

      {/* Error */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {/* Success Result */}
      {result && result.success && (
        <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="w-5 h-5" />
            <span className="font-semibold">Merge Successful!</span>
          </div>
          <div className="text-sm space-y-1">
            <div>Output: {result.output_path}</div>
            <div>Merged {result.merged_count} LoRAs</div>
            <div>File Size: {result.file_size_mb} MB</div>
            <div className="mt-2 font-medium">{result.message}</div>
          </div>
        </div>
      )}
    </div>
  );
}

// ========== Resize LoRA Tab ==========

function ResizeLoRATab() {
  const [inputFile, setInputFile] = useState('');
  const [outputPath, setOutputPath] = useState('');
  const [newDim, setNewDim] = useState(32);
  const [newAlpha, setNewAlpha] = useState(32);
  const [availableFiles, setAvailableFiles] = useState<LoRAFile[]>([]);
  const [availableDims, setAvailableDims] = useState<number[]>([16, 32, 64, 128]);
  const [resizing, setResizing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [outputDir, setOutputDir] = useState<string>('');

  // Load available files and dimensions
  useEffect(() => {
    const loadData = async () => {
      try {
        // Get directories from backend
        const dirsResponse = await utilitiesAPI.getDirectories();
        const loraDir = dirsResponse.output || 'output';
        setOutputDir(loraDir);

        // Load LoRA files
        const filesResponse = await utilitiesAPI.listLoraFiles(loraDir, 'safetensors', 'date');
        if (filesResponse.success) {
          setAvailableFiles(filesResponse.files);
          if (filesResponse.files.length > 0) {
            setInputFile(filesResponse.files[0].path);
          }
        }

        // Load available dimensions
        const dimsResponse = await utilitiesAPI.getResizeDimensions();
        if (dimsResponse.dimensions) {
          setAvailableDims(dimsResponse.dimensions);
        }
      } catch (err) {
        console.error('Failed to load data:', err);
      }
    };
    loadData();
  }, []);

  // Auto-generate output path when input file or dimension changes
  useEffect(() => {
    if (inputFile) {
      const basename = inputFile.replace(/\.safetensors$/, '');
      setOutputPath(`${basename}_dim${newDim}.safetensors`);
    }
  }, [inputFile, newDim]);

  const handleResize = async () => {
    if (!inputFile || !outputPath) {
      setError('Please select an input file and provide an output path');
      return;
    }

    try {
      setResizing(true);
      setError(null);
      setResult(null);

      const response = await utilitiesAPI.resizeLora(inputFile, outputPath, newDim, newAlpha);

      if (response.success) {
        setResult(response);
      } else {
        setError(response.error || 'Resize failed');
      }
    } catch (err: any) {
      setError(err.message || 'Resize failed');
    } finally {
      setResizing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Configuration */}
      <div className="bg-card backdrop-blur-sm border border-border rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold mb-4">Resize Configuration</h2>

        <div className="space-y-4">
          {/* Input File */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">Input LoRA File *</label>
            {availableFiles.length > 0 ? (
              <select
                value={inputFile}
                onChange={(e) => setInputFile(e.target.value)}
                className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
              >
                {availableFiles.map((file) => (
                  <option key={file.path} value={file.path}>
                    {file.name} ({file.size_formatted})
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={inputFile}
                onChange={(e) => setInputFile(e.target.value)}
                className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                placeholder="output/my_lora.safetensors"
              />
            )}
          </div>

          {/* Dimension and Alpha */}
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">New Dimension (Rank) *</label>
              <select
                value={newDim}
                onChange={(e) => setNewDim(parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
              >
                {availableDims.map((dim) => (
                  <option key={dim} value={dim}>
                    {dim}
                  </option>
                ))}
              </select>
              <p className="text-xs text-muted-foreground mt-1">
                Lower = smaller file, less detail
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">New Alpha *</label>
              <input
                type="number"
                value={newAlpha}
                onChange={(e) => setNewAlpha(parseInt(e.target.value) || 0)}
                className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                min="1"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Usually same as dimension
              </p>
            </div>
          </div>

          {/* Output Path */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">Output Path *</label>
            <input
              type="text"
              value={outputPath}
              onChange={(e) => setOutputPath(e.target.value)}
              className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
              placeholder="output/my_lora_dim32.safetensors"
            />
            <p className="text-xs text-muted-foreground mt-1">
              Auto-generated based on input file and dimension
            </p>
          </div>
        </div>

        {/* Info Box */}
        <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-500/30">
          <p className="text-sm font-medium text-foreground mb-2">
            🔧 How Resizing Works
          </p>
          <ul className="text-sm text-foreground space-y-1 list-disc list-inside">
            <li>Uses Derrian's enhanced resize script or Kohya's standard script</li>
            <li>Reduces file size while preserving quality</li>
            <li>Useful for sharing LoRAs or reducing VRAM usage</li>
            <li>Original file is not modified</li>
          </ul>
        </div>
      </div>

      {/* Resize Button */}
      <button
        onClick={handleResize}
        disabled={resizing || !inputFile || !outputPath}
        className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-purple-500 to-indigo-600 text-foreground px-6 py-4 rounded-lg font-semibold hover:from-purple-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
      >
        {resizing ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Resizing...
          </>
        ) : (
          <>
            <Minimize2 className="w-5 h-5" />
            Resize LoRA
          </>
        )}
      </button>

      {/* Error */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {/* Success Result */}
      {result && result.success && (
        <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="w-5 h-5" />
            <span className="font-semibold">Resize Successful!</span>
          </div>
          <div className="text-sm space-y-1">
            <div>Input: {result.input_path}</div>
            <div>Output: {result.output_path}</div>
            <div>Dimension: {result.new_dim} / Alpha: {result.new_alpha}</div>
            <div className="mt-2 font-medium">{result.message}</div>
          </div>
        </div>
      )}
    </div>
  );
}

// ========== Merge Checkpoints Tab ==========

function MergeCheckpointTab() {
  const [availableFiles, setAvailableFiles] = useState<LoRAFile[]>([]);
  const [selectedCheckpoints, setSelectedCheckpoints] = useState<Array<{ path: string; name: string; ratio: number }>>([]);
  const [outputPath, setOutputPath] = useState('');
  const [unetOnly, setUnetOnly] = useState(false);
  const [precision, setPrecision] = useState('float');
  const [savePrecision, setSavePrecision] = useState('fp16');
  const [merging, setMerging] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Load available checkpoint files
  useEffect(() => {
    const loadFiles = async () => {
      try {
        const dirsResponse = await utilitiesAPI.getDirectories();
        const modelDir = dirsResponse.output || 'pretrained_model';

        const filesResponse = await utilitiesAPI.listLoraFiles(modelDir, 'safetensors,ckpt', 'date');
        if (filesResponse.success) {
          // Filter for larger checkpoint files (typically > 100MB)
          const checkpointFiles = filesResponse.files.filter(f => f.size_mb > 100);
          setAvailableFiles(checkpointFiles);
        }
      } catch (err) {
        console.error('Failed to load checkpoint files:', err);
      }
    };
    loadFiles();
  }, []);

  // Auto-generate output path when selection changes
  useEffect(() => {
    if (selectedCheckpoints.length >= 2) {
      const timestamp = new Date().toISOString().split('T')[0];
      const suffix = unetOnly ? '_unet' : '';
      setOutputPath(`pretrained_model/merged${suffix}_${timestamp}.safetensors`);
    }
  }, [selectedCheckpoints.length, unetOnly]);

  const addCheckpointToMerge = (file: LoRAFile) => {
    if (!selectedCheckpoints.find(c => c.path === file.path)) {
      setSelectedCheckpoints([...selectedCheckpoints, { path: file.path, name: file.name, ratio: 1.0 }]);
    }
  };

  const removeCheckpointFromMerge = (path: string) => {
    setSelectedCheckpoints(selectedCheckpoints.filter(c => c.path !== path));
  };

  const updateRatio = (path: string, ratio: number) => {
    setSelectedCheckpoints(
      selectedCheckpoints.map(c =>
        c.path === path ? { ...c, ratio: Math.max(0, Math.min(2, ratio)) } : c
      )
    );
  };

  const handleMerge = async () => {
    if (selectedCheckpoints.length < 2) {
      setError('Please select at least 2 checkpoints to merge');
      return;
    }

    if (!outputPath) {
      setError('Please provide an output path');
      return;
    }

    try {
      setMerging(true);
      setError(null);
      setResult(null);

      const response = await utilitiesAPI.mergeCheckpoint(
        selectedCheckpoints.map(c => ({ path: c.path, ratio: c.ratio })),
        outputPath,
        unetOnly,
        'cpu', // device
        savePrecision,
        precision
      );

      if (response.success) {
        setResult(response);
        setSelectedCheckpoints([]);
      } else {
        setError(response.message || 'Merge failed');
      }
    } catch (err: any) {
      setError(err.message || 'Merge failed');
    } finally {
      setMerging(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Warning Card */}
      <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-400 dark:border-yellow-500/30 rounded-lg p-4">
        <h3 className="font-semibold text-yellow-800 dark:text-yellow-400 mb-2">⚠️ Large File Warning</h3>
        <p className="text-sm text-yellow-700 dark:text-yellow-300">
          Checkpoint merging involves very large files (2-7GB each). Ensure you have sufficient disk space and RAM.
          The merge process may take several minutes depending on your hardware.
        </p>
      </div>

      {/* Merge Options */}
      <div className="bg-card backdrop-blur-sm border border-border rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold text-foreground mb-4">Merge Options</h2>
        <div className="grid md:grid-cols-3 gap-4">
          {/* Precision */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">Computation Precision</label>
            <select
              value={precision}
              onChange={(e) => setPrecision(e.target.value)}
              className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg"
            >
              <option value="float">Float (Recommended)</option>
              <option value="fp16">FP16</option>
              <option value="bf16">BF16</option>
            </select>
          </div>

          {/* Save Precision */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">Save Precision</label>
            <select
              value={savePrecision}
              onChange={(e) => setSavePrecision(e.target.value)}
              className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg"
            >
              <option value="fp16">FP16</option>
              <option value="bf16">BF16</option>
              <option value="float">Float</option>
            </select>
          </div>

          {/* UNet Only */}
          <div className="flex items-end">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={unetOnly}
                onChange={(e) => setUnetOnly(e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm font-medium text-foreground">UNet Only (keep VAE/TE from first model)</span>
            </label>
          </div>
        </div>
      </div>

      {/* Available Checkpoints */}
      <div className="bg-card backdrop-blur-sm border border-border rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold text-foreground mb-4">Available Checkpoints</h2>
        {availableFiles.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            No checkpoint files found (looking for files &gt; 100MB)
          </div>
        ) : (
          <div className="grid md:grid-cols-2 gap-2 max-h-96 overflow-y-auto">
            {availableFiles.map((file) => (
              <button
                key={file.path}
                onClick={() => addCheckpointToMerge(file)}
                disabled={selectedCheckpoints.some(c => c.path === file.path)}
                className="flex items-center justify-between p-3 border border-border rounded-lg hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed text-left"
              >
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-foreground truncate">{file.name}</div>
                  <div className="text-sm text-muted-foreground">{file.size_formatted}</div>
                </div>
                {selectedCheckpoints.some(c => c.path === file.path) && (
                  <CheckCircle className="w-5 h-5 text-green-500 ml-2" />
                )}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Selected Checkpoints */}
      {selectedCheckpoints.length > 0 && (
        <div className="bg-card backdrop-blur-sm border border-border rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold text-foreground mb-4">
            Selected Checkpoints ({selectedCheckpoints.length})
          </h2>
          <div className="space-y-3">
            {selectedCheckpoints.map((checkpoint) => (
              <div key={checkpoint.path} className="flex items-center gap-4 p-3 border border-border rounded-lg">
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-foreground truncate">{checkpoint.name}</div>
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-sm text-muted-foreground whitespace-nowrap">Ratio:</label>
                  <input
                    type="number"
                    value={checkpoint.ratio}
                    onChange={(e) => updateRatio(checkpoint.path, parseFloat(e.target.value) || 0)}
                    step="0.1"
                    min="0"
                    max="2"
                    className="w-20 px-2 py-1 bg-input border border-input text-foreground rounded text-center"
                  />
                </div>
                <button
                  onClick={() => removeCheckpointFromMerge(checkpoint.path)}
                  className="p-2 hover:bg-destructive/10 rounded text-destructive"
                  title="Remove"
                >
                  <XCircle className="w-5 h-5" />
                </button>
              </div>
            ))}
          </div>

          {/* Output Path */}
          <div className="mt-4">
            <label className="block text-sm font-medium text-foreground mb-2">Output Path</label>
            <input
              type="text"
              value={outputPath}
              onChange={(e) => setOutputPath(e.target.value)}
              className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg"
              placeholder="pretrained_model/merged.safetensors"
            />
          </div>
        </div>
      )}

      {/* Merge Button */}
      <button
        onClick={handleMerge}
        disabled={merging || selectedCheckpoints.length < 2 || !outputPath}
        className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-purple-500 to-indigo-600 text-white px-6 py-4 rounded-lg font-semibold hover:from-purple-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
      >
        {merging ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Merging Checkpoints...
          </>
        ) : (
          <>
            <FolderOpen className="w-5 h-5" />
            Merge {selectedCheckpoints.length} Checkpoints
          </>
        )}
      </button>

      {/* Error */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {/* Success Result */}
      {result && result.success && (
        <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="w-5 h-5" />
            <span className="font-semibold">Checkpoint Merge Successful!</span>
          </div>
          <div className="text-sm space-y-1">
            <div>Output: {result.output_path}</div>
            <div>Merged {result.merged_count} checkpoints</div>
            <div>File Size: {result.file_size_mb} MB</div>
            {unetOnly && <div className="text-yellow-700 mt-2">ℹ️ UNet-only merge: VAE and Text Encoder from first model</div>}
            <div className="mt-2 font-medium">{result.message}</div>
          </div>
        </div>
      )}
    </div>
  );
}
