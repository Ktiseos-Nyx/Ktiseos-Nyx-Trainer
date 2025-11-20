'use client';

import { useState, useEffect } from 'react';
import { utilitiesAPI, LoRAFile } from '@/lib/api';
import { Wrench, Upload, FolderOpen, CheckCircle, XCircle, Loader2, Minimize2, Home } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';

export default function UtilitiesPage() {
  const [activeTab, setActiveTab] = useState<'merge' | 'resize' | 'huggingface'>('merge');

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 py-16">
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
            LoRA Utilities
          </h1>
          <p className="text-xl text-gray-300">
            Merge, optimize, and publish your trained LoRAs
          </p>
        </div>

        {/* Tabs */}
        <div className="mb-6">
          <div className="border-b border-slate-700">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setActiveTab('merge')}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'merge'
                    ? 'border-cyan-400 text-cyan-400'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-slate-600'
                }`}
              >
                <FolderOpen className="w-5 h-5 inline mr-2" />
                Merge LoRAs
              </button>
              <button
                onClick={() => setActiveTab('resize')}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'resize'
                    ? 'border-cyan-400 text-cyan-400'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-slate-600'
                }`}
              >
                <Minimize2 className="w-5 h-5 inline mr-2" />
                Resize LoRA
              </button>
              <button
                onClick={() => setActiveTab('huggingface')}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'huggingface'
                    ? 'border-cyan-400 text-cyan-400'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-slate-600'
                }`}
              >
                <Upload className="w-5 h-5 inline mr-2" />
                HuggingFace Upload
              </button>
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'merge' && <MergeLoRATab />}
        {activeTab === 'resize' && <ResizeLoRATab />}
        {activeTab === 'huggingface' && <HuggingFaceTab />}
      </div>
    </div>
  );
}

// ========== Merge LoRAs Tab ==========

function MergeLoRATab() {
  const [mergeType, setMergeType] = useState<'lora-to-lora' | 'lora-to-checkpoint'>('lora-to-lora');

  return (
    <div className="space-y-6">
      {/* Merge Type Selector */}
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold text-white mb-4">Merge Type</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <button
            onClick={() => setMergeType('lora-to-lora')}
            className={`p-4 rounded-lg border-2 transition-all ${
              mergeType === 'lora-to-lora'
                ? 'border-cyan-500 bg-cyan-500/10 text-cyan-400'
                : 'border-slate-600 bg-slate-800/30 text-gray-400 hover:border-slate-500'
            }`}
          >
            <div className="text-center">
              <FolderOpen className="w-8 h-8 mx-auto mb-2" />
              <h3 className="font-semibold">Merge Multiple LoRAs</h3>
              <p className="text-sm mt-1 opacity-80">Combine 2+ LoRAs with custom weights</p>
            </div>
          </button>
          <button
            onClick={() => setMergeType('lora-to-checkpoint')}
            className={`p-4 rounded-lg border-2 transition-all ${
              mergeType === 'lora-to-checkpoint'
                ? 'border-cyan-500 bg-cyan-500/10 text-cyan-400'
                : 'border-slate-600 bg-slate-800/30 text-gray-400 hover:border-slate-500'
            }`}
          >
            <div className="text-center">
              <Upload className="w-8 h-8 mx-auto mb-2" />
              <h3 className="font-semibold">Bake LoRA into Model</h3>
              <p className="text-sm mt-1 opacity-80">Merge LoRA into a base checkpoint</p>
            </div>
          </button>
        </div>
      </div>

      {/* Placeholder Content */}
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold text-white mb-4">
          {mergeType === 'lora-to-lora' ? 'LoRA Merging' : 'LoRA to Checkpoint'}
        </h2>
        <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-6 text-center">
          <p className="text-blue-300 mb-2">🚧 Coming Soon!</p>
          <p className="text-sm text-gray-400">
            {mergeType === 'lora-to-lora'
              ? 'Merge multiple LoRAs together with custom weight ratios using Kohya\'s merge scripts.'
              : 'Bake a LoRA into a base model checkpoint for standalone use without needing the LoRA file.'}
          </p>
          <p className="text-xs text-gray-500 mt-4">
            TODO: Add file selection, weight sliders, merge options, and backend API integration.
          </p>
        </div>
      </div>
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

  // Load available files and dimensions
  useEffect(() => {
    const loadData = async () => {
      try {
        // Load LoRA files
        const filesResponse = await utilitiesAPI.listLoraFiles('/workspace/output', 'safetensors', 'date');
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
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold mb-4">Resize Configuration</h2>

        <div className="space-y-4">
          {/* Input File */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Input LoRA File *</label>
            {availableFiles.length > 0 ? (
              <select
                value={inputFile}
                onChange={(e) => setInputFile(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 text-white rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
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
                className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 text-white rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                placeholder="/workspace/output/my_lora.safetensors"
              />
            )}
          </div>

          {/* Dimension and Alpha */}
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">New Dimension (Rank) *</label>
              <select
                value={newDim}
                onChange={(e) => setNewDim(parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 text-white rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
              >
                {availableDims.map((dim) => (
                  <option key={dim} value={dim}>
                    {dim}
                  </option>
                ))}
              </select>
              <p className="text-xs text-gray-500 mt-1">
                Lower = smaller file, less detail
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">New Alpha *</label>
              <input
                type="number"
                value={newAlpha}
                onChange={(e) => setNewAlpha(parseInt(e.target.value) || 0)}
                className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 text-white rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                min="1"
              />
              <p className="text-xs text-gray-500 mt-1">
                Usually same as dimension
              </p>
            </div>
          </div>

          {/* Output Path */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Output Path *</label>
            <input
              type="text"
              value={outputPath}
              onChange={(e) => setOutputPath(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 text-white rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
              placeholder="/workspace/output/my_lora_dim32.safetensors"
            />
            <p className="text-xs text-gray-500 mt-1">
              Auto-generated based on input file and dimension
            </p>
          </div>
        </div>

        {/* Info Box */}
        <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
          <p className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-2">
            🔧 How Resizing Works
          </p>
          <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1 list-disc list-inside">
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
        className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-purple-500 to-indigo-600 text-white px-6 py-4 rounded-lg font-semibold hover:from-purple-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
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

// ========== HuggingFace Tab ==========

function HuggingFaceTab() {
  const [uploadType, setUploadType] = useState<'lora' | 'dataset'>('lora');
  const [hfToken, setHfToken] = useState('');
  const [owner, setOwner] = useState('');
  const [repoName, setRepoName] = useState('');
  const [repoType, setRepoType] = useState('model');
  const [remoteFolder, setRemoteFolder] = useState('');
  const [commitMessage, setCommitMessage] = useState('Upload via Ktiseos-Nyx-Trainer 🤗');
  const [createPR, setCreatePR] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [availableFiles, setAvailableFiles] = useState<LoRAFile[]>([]);
  const [tokenValid, setTokenValid] = useState<boolean | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [datasetDirectory, setDatasetDirectory] = useState('/workspace/datasets');

  // Load available files based on upload type
  useEffect(() => {
    const loadFiles = async () => {
      try {
        if (uploadType === 'lora') {
          const response = await utilitiesAPI.listLoraFiles('/workspace/output', 'safetensors', 'date');
          if (response.success) {
            setAvailableFiles(response.files);
          }
        } else {
          // For datasets, we'll browse the dataset directory
          // This will need a different API endpoint or file listing
          setAvailableFiles([]);
        }
      } catch (err) {
        console.error('Failed to load files:', err);
      }
    };
    loadFiles();
    setSelectedFiles([]); // Clear selections when switching type
  }, [uploadType]);

  // Validate token
  const handleValidateToken = async () => {
    if (!hfToken.trim()) {
      setTokenValid(null);
      return;
    }

    try {
      const response = await utilitiesAPI.validateHfToken(hfToken);
      setTokenValid(response.valid);

      if (response.valid && response.username) {
        setOwner(response.username);
      }
    } catch (err) {
      setTokenValid(false);
    }
  };

  // Upload to HuggingFace
  const handleUpload = async () => {
    if (!hfToken || !owner || !repoName || selectedFiles.length === 0) {
      setError('Please fill in all required fields and select at least one file');
      return;
    }

    try {
      setUploading(true);
      setError(null);
      setUploadResult(null);

      const response = await utilitiesAPI.uploadToHuggingFace({
        hf_token: hfToken,
        owner,
        repo_name: repoName,
        repo_type: repoType,
        selected_files: selectedFiles,
        remote_folder: remoteFolder,
        commit_message: commitMessage,
        create_pr: createPR,
      });

      if (response.success) {
        setUploadResult(response);
        setSelectedFiles([]);
      } else {
        setError(response.error || 'Upload failed');
      }
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const toggleFileSelection = (filePath: string) => {
    setSelectedFiles((prev) =>
      prev.includes(filePath)
        ? prev.filter((f) => f !== filePath)
        : [...prev, filePath]
    );
  };

  return (
    <div className="space-y-6">
      {/* Upload Type Selector */}
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold text-white mb-4">What would you like to upload?</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <button
            onClick={() => setUploadType('lora')}
            className={`p-4 rounded-lg border-2 transition-all ${
              uploadType === 'lora'
                ? 'border-cyan-500 bg-cyan-500/10 text-cyan-400'
                : 'border-slate-600 bg-slate-800/30 text-gray-400 hover:border-slate-500'
            }`}
          >
            <div className="text-center">
              <Minimize2 className="w-8 h-8 mx-auto mb-2" />
              <h3 className="font-semibold">LoRA Models</h3>
              <p className="text-sm mt-1 opacity-80">Upload trained LoRA .safetensors files</p>
            </div>
          </button>
          <button
            onClick={() => setUploadType('dataset')}
            className={`p-4 rounded-lg border-2 transition-all ${
              uploadType === 'dataset'
                ? 'border-cyan-500 bg-cyan-500/10 text-cyan-400'
                : 'border-slate-600 bg-slate-800/30 text-gray-400 hover:border-slate-500'
            }`}
          >
            <div className="text-center">
              <FolderOpen className="w-8 h-8 mx-auto mb-2" />
              <h3 className="font-semibold">Datasets</h3>
              <p className="text-sm mt-1 opacity-80">Upload training datasets with images & captions</p>
            </div>
          </button>
        </div>
      </div>

      {/* Token & Repository Config */}
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold text-white mb-4">HuggingFace Configuration</h2>

        {/* Token */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            HuggingFace Token (Write Access) *
          </label>
          <div className="flex gap-2">
            <input
              type="password"
              value={hfToken}
              onChange={(e) => {
                setHfToken(e.target.value);
                setTokenValid(null);
              }}
              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              placeholder="hf_..."
            />
            <button
              onClick={handleValidateToken}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
            >
              Validate
            </button>
          </div>

          {tokenValid !== null && (
            <div className={`mt-2 flex items-center gap-2 text-sm ${tokenValid ? 'text-green-600' : 'text-red-600'}`}>
              {tokenValid ? <CheckCircle className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
              {tokenValid ? 'Token is valid!' : 'Invalid token'}
            </div>
          )}

          <p className="mt-2 text-xs text-gray-500">
            Get your token at{' '}
            <a href="https://huggingface.co/settings/tokens" target="_blank" className="text-blue-500 hover:underline">
              huggingface.co/settings/tokens
            </a>
          </p>
        </div>

        {/* Repository Config */}
        <div className="grid md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Owner *</label>
            <input
              type="text"
              value={owner}
              onChange={(e) => setOwner(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 text-white rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
              placeholder="your-username"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Repository *</label>
            <input
              type="text"
              value={repoName}
              onChange={(e) => setRepoName(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 text-white rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
              placeholder="my-loras"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Type</label>
            <select
              value={repoType}
              onChange={(e) => setRepoType(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 text-white rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
            >
              <option value="model">Model</option>
              <option value="dataset">Dataset</option>
              <option value="space">Space</option>
            </select>
          </div>
        </div>

        {/* Optional Settings */}
        <div className="mt-4 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Remote Folder (Optional)</label>
            <input
              type="text"
              value={remoteFolder}
              onChange={(e) => setRemoteFolder(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 text-white rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
              placeholder="models/v1"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Commit Message</label>
            <textarea
              value={commitMessage}
              onChange={(e) => setCommitMessage(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 text-white rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
              rows={2}
            />
          </div>

          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={createPR}
              onChange={(e) => setCreatePR(e.target.checked)}
              className="w-4 h-4 text-blue-600 rounded"
            />
            <span className="text-sm">Create Pull Request</span>
          </label>
        </div>
      </div>

      {/* File Selection */}
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold text-white mb-4">
          Select {uploadType === 'lora' ? 'LoRA Files' : 'Dataset Files'} ({selectedFiles.length} selected)
        </h2>

        {uploadType === 'dataset' && (
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">Dataset Directory</label>
            <input
              type="text"
              value={datasetDirectory}
              onChange={(e) => setDatasetDirectory(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 text-white rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
              placeholder="/workspace/datasets/my-dataset"
            />
            <p className="text-xs text-gray-500 mt-1">
              Specify the dataset folder to upload (will upload all contents)
            </p>
          </div>
        )}

        {uploadType === 'lora' && availableFiles.length === 0 ? (
          <div className="text-center py-12">
            <FolderOpen className="w-16 h-16 mx-auto text-gray-300 mb-4" />
            <p className="text-gray-400">No LoRA files found</p>
          </div>
        ) : uploadType === 'dataset' ? (
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <p className="text-sm text-blue-300">
              📁 Dataset upload will include all files from: <span className="font-mono text-cyan-400">{datasetDirectory}</span>
            </p>
            <p className="text-xs text-gray-400 mt-2">
              This includes images, caption files (.txt), and any other files in the directory.
            </p>
          </div>
        ) : (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {availableFiles.map((file) => (
              <label
                key={file.path}
                className="flex items-start gap-3 p-3 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={selectedFiles.includes(file.path)}
                  onChange={() => toggleFileSelection(file.path)}
                  className="mt-1 w-4 h-4 text-blue-600 rounded"
                />
                <div className="flex-1">
                  <div className="font-semibold">{file.name}</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    {file.size_formatted}
                  </div>
                </div>
              </label>
            ))}
          </div>
        )}
      </div>

      {/* Upload Button */}
      <button
        onClick={handleUpload}
        disabled={uploading || selectedFiles.length === 0}
        className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white px-6 py-4 rounded-lg font-semibold hover:from-green-600 hover:to-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
      >
        {uploading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Uploading...
          </>
        ) : (
          <>
            <Upload className="w-5 h-5" />
            Upload to HuggingFace
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
      {uploadResult && uploadResult.success && (
        <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="w-5 h-5" />
            <span className="font-semibold">Upload Successful!</span>
          </div>
          <div className="text-sm space-y-1">
            <div>Repository: {uploadResult.repo_id}</div>
            <div>Uploaded: {uploadResult.uploaded_files.length} files</div>
            {uploadResult.failed_files.length > 0 && (
              <div className="text-red-600">Failed: {uploadResult.failed_files.length} files</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
