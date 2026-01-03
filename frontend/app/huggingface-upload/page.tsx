'use client';

import { useState, useEffect } from 'react';
import { utilitiesAPI, LoRAFile } from '@/lib/api';
import { Upload, FolderOpen, CheckCircle, XCircle, Loader2, Minimize2, Home } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';

export default function HuggingFaceUploadPage() {
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
  const [datasetDirectory, setDatasetDirectory] = useState('');

  // Load available files based on upload type
  useEffect(() => {
    const loadFiles = async () => {
      try {
        // Get directories from backend
        const dirsResponse = await utilitiesAPI.getDirectories();
        const loraDir = dirsResponse.output || 'output';
        const datasetsDir = dirsResponse.datasets || 'datasets';
        setDatasetDirectory(datasetsDir);

        if (uploadType === 'lora') {
          const response = await utilitiesAPI.listLoraFiles(loraDir, 'safetensors', 'date');
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
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'File Management', href: '/files', icon: <FolderOpen className="w-4 h-4" /> },
            { label: 'HuggingFace Upload', icon: <Upload className="w-4 h-4" /> },
          ]}
        />

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-orange-400 via-yellow-400 to-orange-400 bg-clip-text text-transparent">
            HuggingFace Upload
          </h1>
          <p className="text-xl text-muted-foreground">
            Upload trained LoRAs or datasets to HuggingFace Hub
          </p>
        </div>

        <div className="space-y-6">
          {/* Upload Type Selector */}
          <div className="bg-card backdrop-blur-sm border border-border rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-foreground mb-4">What would you like to upload?</h2>
            <div className="grid md:grid-cols-2 gap-4">
              <button
                onClick={() => setUploadType('lora')}
                className={`p-4 rounded-lg border-2 transition-all ${
                  uploadType === 'lora'
                    ? 'border-cyan-500 bg-cyan-500/10 text-cyan-400'
                    : 'border-input bg-card/50 text-muted-foreground hover:border-slate-500'
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
                    : 'border-input bg-card/50 text-muted-foreground hover:border-slate-500'
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
          <div className="bg-card backdrop-blur-sm border border-border rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-foreground mb-4">HuggingFace Configuration</h2>

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
                  className="flex-1 px-3 py-2 bg-input border border-input text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500"
                  placeholder="hf_..."
                />
                <button
                  onClick={handleValidateToken}
                  className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg"
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

              <p className="mt-2 text-xs text-muted-foreground">
                Get your token at{' '}
                <a href="https://huggingface.co/settings/tokens" target="_blank" className="text-cyan-500 hover:underline">
                  huggingface.co/settings/tokens
                </a>
              </p>
            </div>

            {/* Repository Config */}
            <div className="grid md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">Owner *</label>
                <input
                  type="text"
                  value={owner}
                  onChange={(e) => setOwner(e.target.value)}
                  className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                  placeholder="your-username"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">Repository *</label>
                <input
                  type="text"
                  value={repoName}
                  onChange={(e) => setRepoName(e.target.value)}
                  className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                  placeholder="my-loras"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">Type</label>
                <select
                  value={repoType}
                  onChange={(e) => setRepoType(e.target.value)}
                  className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
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
                <label className="block text-sm font-medium text-foreground mb-2">Remote Folder (Optional)</label>
                <input
                  type="text"
                  value={remoteFolder}
                  onChange={(e) => setRemoteFolder(e.target.value)}
                  className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                  placeholder="models/v1"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">Commit Message</label>
                <textarea
                  value={commitMessage}
                  onChange={(e) => setCommitMessage(e.target.value)}
                  className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                  rows={2}
                />
              </div>

              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={createPR}
                  onChange={(e) => setCreatePR(e.target.checked)}
                  className="w-4 h-4 text-cyan-600 rounded"
                />
                <span className="text-sm">Create Pull Request</span>
              </label>
            </div>
          </div>

          {/* File Selection */}
          <div className="bg-card backdrop-blur-sm border border-border rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-foreground mb-4">
              Select {uploadType === 'lora' ? 'LoRA Files' : 'Dataset Files'} ({selectedFiles.length} selected)
            </h2>

            {uploadType === 'dataset' && (
              <div className="mb-4">
                <label className="block text-sm font-medium text-foreground mb-2">Dataset Directory</label>
                <input
                  type="text"
                  value={datasetDirectory}
                  onChange={(e) => setDatasetDirectory(e.target.value)}
                  className="w-full px-3 py-2 bg-input border border-input text-foreground rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                  placeholder="datasets/my-dataset"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Specify the dataset folder to upload (will upload all contents)
                </p>
              </div>
            )}

            {uploadType === 'lora' && availableFiles.length === 0 ? (
              <div className="text-center py-12">
                <FolderOpen className="w-16 h-16 mx-auto text-foreground mb-4" />
                <p className="text-muted-foreground">No LoRA files found</p>
              </div>
            ) : uploadType === 'dataset' ? (
              <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-4">
                <p className="text-sm text-cyan-400">
                  📁 Dataset upload will include all files from: <span className="font-mono text-cyan-300">{datasetDirectory}</span>
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  This includes images, caption files (.txt), and any other files in the directory.
                </p>
              </div>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {availableFiles.map((file) => (
                  <label
                    key={file.path}
                    className="flex items-start gap-3 p-3 border border-border rounded-lg hover:bg-accent cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={selectedFiles.includes(file.path)}
                      onChange={() => toggleFileSelection(file.path)}
                      className="mt-1 w-4 h-4 text-cyan-600 rounded"
                    />
                    <div className="flex-1">
                      <div className="font-semibold">{file.name}</div>
                      <div className="text-sm text-muted-foreground">
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
      </div>
    </div>
  );
}
