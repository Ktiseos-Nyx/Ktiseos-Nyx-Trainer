'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Upload,
  Image as ImageIcon,
  CheckCircle,
  XCircle,
  Loader,
  FolderPlus,
  Link as LinkIcon,
  FileArchive,
  X,
  RefreshCw,
  Download,
  AlertTriangle,
} from 'lucide-react';
import { zipSync } from 'fflate';
import { toast } from 'sonner';
import { datasetAPI } from '@/lib/api';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';

interface UploadedFile {
  file: File;
  status: 'pending' | 'uploading' | 'success' | 'error';
  progress: number;
  error?: string;
  preview?: string; // Blob URL for image preview (revoked on cleanup)
}

export default function DatasetUploader() {
  // ========== State Management ==========
  const [activeTab, setActiveTab] = useState('direct');

  // Existing datasets check
  const [existingDatasets, setExistingDatasets] = useState<string[]>([]);
  const [datasetExists, setDatasetExists] = useState(false);

  // Direct Upload State
  const [datasetName, setDatasetName] = useState('my_dataset');
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const filesRef = useRef<UploadedFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [zipUploadProgress, setZipUploadProgress] = useState(0);

  // URL/ZIP Download State
  const [projectName, setProjectName] = useState('');
  const [datasetUrl, setDatasetUrl] = useState('');
  const [downloading, setDownloading] = useState(false);

  // Folder Creation State
  const [folderName, setFolderName] = useState('');
  const [folderRepeats, setFolderRepeats] = useState(10);
  const [creatingFolder, setCreatingFolder] = useState(false);

  // Check states for other tabs
  const [projectExists, setProjectExists] = useState(false);
  const [folderExists, setFolderExists] = useState(false);

  // Load existing datasets on mount
  useEffect(() => {
    const loadExistingDatasets = async () => {
      try {
        const data = await datasetAPI.list();
        setExistingDatasets((data.datasets || []).map(d => d.name));
      } catch (err) {
        console.error('Failed to load existing datasets:', err);
      }
    };
    loadExistingDatasets();
  }, []);

  // Check if dataset name exists
  useEffect(() => {
    setDatasetExists(existingDatasets.includes(datasetName.trim()));
  }, [datasetName, existingDatasets]);

  // Check if project name exists (URL/ZIP tab)
  useEffect(() => {
    setProjectExists(existingDatasets.includes(projectName.trim()));
  }, [projectName, existingDatasets]);

  // Check if folder name exists (Create Folder tab)
  useEffect(() => {
    const kohyaFolderName = folderName.trim() ? `${folderRepeats}_${folderName.trim()}` : '';
    setFolderExists(kohyaFolderName ? existingDatasets.includes(kohyaFolderName) : false);
  }, [folderName, folderRepeats, existingDatasets]);

  // Keep filesRef in sync so the unmount cleanup always sees the latest files
  useEffect(() => { filesRef.current = files; }, [files]);

  // ========== Direct Upload Handlers ==========

  // Handle file drop
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadedFile[] = acceptedFiles.map((file) => ({
      file,
      status: 'pending',
      progress: 0,
      preview: file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined,
    }));

    setFiles((prev) => [...prev, ...newFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp', '.bmp'],
      'application/zip': ['.zip'],
    },
    multiple: true,
  });

  // Check if Remote GPU Mode is enabled in user settings
  const isRemoteGPU = (): boolean => {
    try {
      const stored = localStorage.getItem('ktiseos-nyx-settings');
      if (stored) {
        return JSON.parse(stored).remoteGPU === true;
      }
    } catch { /* ignore */ }
    return false;
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      toast.warning('No files to upload');
      return;
    }

    if (!datasetName.trim()) {
      toast.warning('Please enter a dataset name');
      return;
    }

    // Check if dataset exists and confirm
    if (datasetExists) {
      if (!confirm(`Dataset "${datasetName}" already exists!\n\nFiles will be added to the existing dataset. Continue?`)) {
        return;
      }
    }

    setUploading(true);

    try {
      const imageFiles = files.filter(f => f.file.type.startsWith('image/'));

      if (isRemoteGPU()) {
        // Remote GPU Mode: zip all images and send as single request
        await handleUploadZipped(imageFiles);
      } else {
        // Local mode: upload individually with per-file progress
        await handleBatchUpload(imageFiles);
      }

      toast.success('Upload complete!');

      // Reload existing datasets list
      try {
        const data = await datasetAPI.list();
        setExistingDatasets((data.datasets || []).map(d => d.name));
      } catch (err) {
        console.error('Failed to reload datasets:', err);
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') return;
      console.error('Upload failed:', err);
      toast.error(`Upload failed: ${err}`);
    } finally {
      setUploading(false);
    }
  };

  // Upload all files in a single batch request (avoids per-file round-trip overhead)
  const handleBatchUpload = async (imageFiles: UploadedFile[]) => {
    if (imageFiles.length === 0) return;

    const imageFileSet = new Set(imageFiles.map(f => f.file));

    // Mark all as uploading at once
    setFiles(prev =>
      prev.map(f =>
        imageFileSet.has(f.file) ? { ...f, status: 'uploading' } : f
      )
    );

    try {
      await datasetAPI.uploadBatch(imageFiles.map(f => f.file), datasetName);

      setFiles(prev =>
        prev.map(f =>
          imageFileSet.has(f.file)
            ? { ...f, status: 'success', progress: 100 }
            : f
        )
      );
    } catch (err) {
      setFiles(prev =>
        prev.map(f =>
          imageFileSet.has(f.file)
            ? { ...f, status: 'error', error: String(err) }
            : f
        )
      );
      throw err;
    }
  };

  // ========== Chunked ZIP Upload ==========

  const CHUNK_SIZE = 10 * 1024 * 1024; // 10 MB per chunk
  const MAX_RETRIES = 3;

  /**
   * Upload a ZIP file (File or in-memory Blob) in 10 MB chunks via the
   * upload-chunk-init → upload-chunk × N → upload-chunk-finalize route trio.
   * Progress is reported as chunk index / totalChunks.
   * Each chunk is retried up to MAX_RETRIES times with exponential backoff.
   */
  const handleChunkedZipUpload = async (
    zipFile: File | Blob,
    fileName: string,
    targetDatasetName: string,
  ): Promise<{ success: boolean; extracted?: number; errors?: string[] }> => {
    const uploadId = crypto.randomUUID();
    const totalChunks = Math.ceil(zipFile.size / CHUNK_SIZE);

    const initRes = await fetch('/api/dataset/upload-chunk-init', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uploadId, fileName, totalChunks }),
    });
    if (!initRes.ok) throw new Error('Failed to initialise chunked upload');

    setZipUploadProgress(0);

    for (let i = 0; i < totalChunks; i++) {
      let retries = 0;
      while (retries < MAX_RETRIES) {
        try {
          const start = i * CHUNK_SIZE;
          const chunk = zipFile.slice(start, start + CHUNK_SIZE);
          const form = new FormData();
          form.append('chunk', chunk, `part_${i}`);
          form.append('uploadId', uploadId);
          form.append('index', String(i));
          const res = await fetch('/api/dataset/upload-chunk', {
            method: 'POST',
            body: form,
            keepalive: true, // keeps Cloudflared/Caddy tunnel alive during slow chunks
          });
          if (!res.ok) throw new Error(`Chunk ${i} rejected by server`);
          setZipUploadProgress(Math.round(((i + 1) / totalChunks) * 100));
          break;
        } catch (err) {
          retries++;
          if (retries === MAX_RETRIES) throw new Error(`Chunk ${i} failed after ${MAX_RETRIES} retries: ${err}`);
          await new Promise(r => setTimeout(r, 1500 * retries));
        }
      }
    }

    const finalRes = await fetch('/api/dataset/upload-chunk-finalize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uploadId, fileName, datasetName: targetDatasetName }),
    });
    setZipUploadProgress(0);

    if (!finalRes.ok) {
      const errData = await finalRes.json().catch(() => ({})) as { detail?: string; error?: string };
      throw new Error(errData.detail || errData.error || 'Upload finalisation failed');
    }

    return finalRes.json() as Promise<{ success: boolean; extracted?: number; errors?: string[] }>;
  };

  // Remote GPU mode: zip images in memory, then send via chunked upload
  const handleUploadZipped = async (imageFiles: UploadedFile[]) => {
    setFiles(prev =>
      prev.map(f =>
        imageFiles.some(img => img.file === f.file) ? { ...f, status: 'uploading' } : f
      )
    );

    const zipData: { [key: string]: [Uint8Array, { level: 6 }] } = {};
    for (const fileObj of imageFiles) {
      const buf = await fileObj.file.arrayBuffer();
      zipData[fileObj.file.name] = [new Uint8Array(buf), { level: 6 as const }];
    }
    const zipped = zipSync(zipData);
    const zipBlob = new Blob([zipped.buffer as ArrayBuffer], { type: 'application/zip' });

    const result = await handleChunkedZipUpload(zipBlob, `${datasetName}.zip`, datasetName);
    if (!result.success) {
      throw new Error(`No files extracted from ZIP (got ${result.extracted || 0} files)`);
    }

    setFiles(prev =>
      prev.map(f =>
        imageFiles.some(img => img.file === f.file) ? { ...f, status: 'success', progress: 100 } : f
      )
    );
  };

  // Direct ZIP drop: send via chunked upload
  const handleUploadZip = async () => {
    const zipFiles = files.filter(f => f.file.name.endsWith('.zip'));

    if (zipFiles.length === 0) { toast.warning('No ZIP files selected'); return; }
    if (!datasetName.trim()) { toast.warning('Please enter a dataset name'); return; }

    if (datasetExists) {
      if (!confirm(`Dataset "${datasetName}" already exists!\n\nZIP contents will be extracted to the existing dataset. Continue?`)) return;
    }

    const zipFile = zipFiles[0].file;
    if (!zipFile || zipFile.size === 0) { toast.error('ZIP file is empty or not loaded yet'); return; }

    setUploading(true);
    try {
      const result = await handleChunkedZipUpload(zipFile, zipFile.name, datasetName);
      if (result.success) {
        toast.success(`ZIP uploaded! Extracted ${result.extracted} images`, {
          description: result.errors && result.errors.length > 0 ? `Errors: ${result.errors.join(', ')}` : undefined,
        });
        revokePreviewUrls(files);
        setFiles([]);
        try {
          const data = await datasetAPI.list();
          setExistingDatasets((data.datasets || []).map(d => d.name));
        } catch (err) {
          console.error('Failed to reload datasets:', err);
        }
      } else {
        throw new Error(`No files extracted from ZIP (got ${result.extracted || 0} files)`);
      }
    } catch (err) {
      toast.error(`ZIP upload failed: ${err}`);
    } finally {
      setUploading(false);
      setZipUploadProgress(0);
    }
  };

  // Revoke blob URLs to prevent memory leaks
  const revokePreviewUrls = useCallback((filesToRevoke: UploadedFile[]) => {
    filesToRevoke.forEach(f => {
      if (f.preview) URL.revokeObjectURL(f.preview);
    });
  }, []);

  // Revoke blob URLs on unmount — filesRef.current always holds the latest list
  useEffect(() => {
    return () => { revokePreviewUrls(filesRef.current); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Clear files
  const handleClear = () => {
    revokePreviewUrls(files);
    setFiles([]);
  };

  // Reset upload state
  const handleReset = () => {
    revokePreviewUrls(files);
    setFiles([]);
    setUploading(false);
  };

  // ========== URL/ZIP Download Handlers ==========

  const handleUrlDownload = async () => {
    if (!projectName.trim()) {
      toast.warning('Please enter a project name');
      return;
    }

    if (!datasetUrl.trim()) {
      toast.warning('Please enter a dataset URL or path');
      return;
    }

    // Check if dataset exists and confirm
    if (projectExists) {
      if (!confirm(`⚠️ Dataset "${projectName}" already exists!\n\nDownloaded files will be added to the existing dataset. Continue?`)) {
        return;
      }
    }

    setDownloading(true);

    try {
      const response = await fetch('/api/dataset/download-url', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url: datasetUrl,
          dataset_name: projectName,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Download failed');
      }

      const result = await response.json();

      if (result.success) {
        const fileCount = result.files.length;
        toast.success(`Downloaded ${fileCount} file(s) to datasets/${projectName}`, {
          description: result.errors.length > 0 ? `Errors: ${result.errors.join(', ')}` : undefined,
        });
        setDatasetUrl('');

        // Reload existing datasets list
        try {
          const data = await datasetAPI.list();
          setExistingDatasets((data.datasets || []).map(d => d.name));
        } catch (err) {
          console.error('Failed to reload datasets:', err);
        }
      } else {
        throw new Error('Download failed');
      }
    } catch (err) {
      toast.error(`Download failed: ${err}`);
    } finally {
      setDownloading(false);
    }
  };

  // ========== Folder Creation Handlers ==========

  const handleCreateFolder = async () => {
    if (!folderName.trim()) {
      toast.warning('Please enter a folder name');
      return;
    }

    const kohyaFolderName = `${folderRepeats}_${folderName}`;

    // Check if folder already exists
    if (folderExists) {
      toast.warning(`Folder "${kohyaFolderName}" already exists`, {
        description: 'Please choose a different name or modify it in the Direct Upload tab.',
      });
      return;
    }

    setCreatingFolder(true);

    try {
      await datasetAPI.create(kohyaFolderName);
      setDatasetName(kohyaFolderName);
      toast.success(`Created dataset folder: ${kohyaFolderName}`);

      // Reload existing datasets list
      try {
        const data = await datasetAPI.list();
        setExistingDatasets((data.datasets || []).map(d => d.name));
      } catch (err) {
        console.error('Failed to reload datasets:', err);
      }

      // Switch to direct upload tab
      setActiveTab('direct');
    } catch (err) {
      toast.error(`Failed to create folder: ${err}`);
    } finally {
      setCreatingFolder(false);
    }
  };

  // ========== Status Counts ==========
  const successCount = files.filter((f) => f.status === 'success').length;
  const errorCount = files.filter((f) => f.status === 'error').length;
  const pendingCount = files.filter((f) => f.status === 'pending').length;
  const zipCount = files.filter((f) => f.file.name.endsWith('.zip')).length;
  const imageCount = files.filter((f) => f.file.type.startsWith('image/')).length;

  return (
    <div className="bg-card/50 backdrop-blur-sm rounded-lg border border-border p-6 shadow-sm">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-foreground mb-2">Dataset Upload</h2>
        <p className="text-muted-foreground">
          Choose your preferred method to set up your training dataset
        </p>
      </div>

      {/* Upload Method Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-3 bg-muted border border-border">
          <TabsTrigger value="direct" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
            <Upload className="w-4 h-4 mr-2" />
            Direct Upload
          </TabsTrigger>
          <TabsTrigger value="url" className="data-[state=active]:bg-purple-600 data-[state=active]:text-white">
            <LinkIcon className="w-4 h-4 mr-2" />
            URL/ZIP
          </TabsTrigger>
          <TabsTrigger value="folder" className="data-[state=active]:bg-green-600 data-[state=active]:text-white">
            <FolderPlus className="w-4 h-4 mr-2" />
            Create Folder
          </TabsTrigger>
        </TabsList>

        {/* ========== Direct Upload Tab ========== */}
        <TabsContent value="direct" className="space-y-6">
          {/* Dataset Name Input */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Dataset Name
            </label>
            <input
              type="text"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              className={`w-full px-4 py-2 bg-input border rounded-lg text-foreground focus:ring-2 focus:border-transparent ${
                datasetExists
                  ? 'border-yellow-500 focus:ring-yellow-500'
                  : 'border-input focus:ring-blue-500'
              }`}
              placeholder="my_awesome_dataset"
              disabled={uploading}
            />
            {datasetExists ? (
              <p className="text-xs text-yellow-500 mt-1 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" />
                ⚠️ Dataset exists - files will be added to existing dataset
              </p>
            ) : (
              <p className="text-xs text-muted-foreground mt-1">
                📁 Files will be uploaded to: datasets/{datasetName}
              </p>
            )}
          </div>

          {/* Drop Zone */}
          <div
            {...getRootProps()}
            className={`
              border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-all
              ${
                isDragActive
                  ? 'border-blue-500 bg-blue-500/10'
                  : 'border-border hover:border-blue-400 hover:bg-accent/50'
              }
            `}
          >
            <input {...getInputProps()} />

            <Upload
              className={`w-16 h-16 mx-auto mb-4 ${
                isDragActive ? 'text-blue-500 animate-bounce' : 'text-muted-foreground'
              }`}
            />

            {isDragActive ? (
              <p className="text-lg font-semibold text-blue-600 dark:text-blue-400">
                Drop files here!
              </p>
            ) : (
              <>
                <p className="text-lg font-semibold text-foreground mb-2">
                  Drag & drop images or ZIP files
                </p>
                <p className="text-sm text-muted-foreground">
                  or click to browse files
                </p>
                <p className="text-xs text-muted-foreground/70 mt-2">
                  Supports: PNG, JPG, JPEG, WebP, BMP, ZIP
                </p>
              </>
            )}
          </div>

          {/* File List */}
          {files.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold text-foreground">
                  Files ({files.length})
                  {imageCount > 0 && ` · ${imageCount} images`}
                  {zipCount > 0 && ` · ${zipCount} ZIP`}
                </h3>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={handleReset}
                    title="Force reset - always works even if upload is stuck"
                    className="text-yellow-600 dark:text-yellow-400 hover:text-yellow-700 dark:hover:text-yellow-300 gap-1 h-auto py-0 px-1"
                  >
                    <RefreshCw className="w-3 h-3" />
                    Reset {uploading && '(Force)'}
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={handleClear}
                    disabled={uploading}
                    className="text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 gap-1 h-auto py-0 px-1"
                  >
                    <X className="w-3 h-3" />
                    Clear All
                  </Button>
                </div>
              </div>

              {/* Status Summary */}
              <div className="flex gap-4 mb-4 text-sm">
                {pendingCount > 0 && (
                  <span className="text-muted-foreground">
                    ⏳ Pending: {pendingCount}
                  </span>
                )}
                {successCount > 0 && (
                  <span className="text-green-600 dark:text-green-400">
                    ✓ Success: {successCount}
                  </span>
                )}
                {errorCount > 0 && (
                  <span className="text-red-600 dark:text-red-400">
                    ✗ Error: {errorCount}
                  </span>
                )}
              </div>

              {/* File Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 max-h-96 overflow-y-auto p-1">
                {files.map((fileObj, index) => {
                  const isZip = fileObj.file.name.endsWith('.zip');

                  return (
                    <div
                      key={index}
                      className="relative group rounded-lg overflow-hidden border-2 border-slate-700 bg-slate-800"
                    >
                      {/* Image/ZIP Preview */}
                      {isZip ? (
                        <div className="w-full h-32 flex items-center justify-center bg-slate-800">
                          <FileArchive className="w-12 h-12 text-purple-400" />
                        </div>
                      ) : (
                        <img
                          src={fileObj.preview}
                          alt={fileObj.file.name}
                          className="w-full h-32 object-cover"
                          loading="lazy"
                        />
                      )}

                      {/* Filename Overlay */}
                      <div className="absolute inset-0 bg-black/70 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity p-2">
                        <span className="text-white text-xs text-center break-all">
                          {fileObj.file.name}
                        </span>
                      </div>

                      {/* Status Badge */}
                      <div className="absolute top-2 right-2">
                        {fileObj.status === 'pending' && (
                          <div className="bg-gray-500 text-white p-1 rounded-full">
                            <ImageIcon className="w-4 h-4" />
                          </div>
                        )}
                        {fileObj.status === 'uploading' && (
                          <div className="bg-blue-500 text-white p-1 rounded-full animate-spin">
                            <Loader className="w-4 h-4" />
                          </div>
                        )}
                        {fileObj.status === 'success' && (
                          <div className="bg-green-500 text-white p-1 rounded-full">
                            <CheckCircle className="w-4 h-4" />
                          </div>
                        )}
                        {fileObj.status === 'error' && (
                          <div className="bg-red-500 text-white p-1 rounded-full">
                            <XCircle className="w-4 h-4" />
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3">
            {zipCount > 0 && (
              <Button
                type="button"
                onClick={handleUploadZip}
                disabled={uploading || zipCount === 0}
                className="flex-1 gap-2 bg-gradient-to-r from-purple-500 to-purple-600 text-white hover:from-purple-600 hover:to-purple-700"
              >
                <FileArchive className="w-5 h-5" />
                {uploading ? 'Uploading...' : `Upload ${zipCount} ZIP`}
              </Button>
            )}

            {imageCount > 0 && (
              <Button
                type="button"
                onClick={handleUpload}
                disabled={uploading || imageCount === 0}
                className="flex-1 gap-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white hover:from-blue-600 hover:to-indigo-700"
              >
                <Upload className="w-5 h-5" />
                {uploading ? 'Uploading...' : `Upload ${imageCount} Images`}
              </Button>
            )}
          </div>

          {/* Upload Progress */}
          {uploading && zipUploadProgress > 0 && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Uploading...</span>
                <span>{zipUploadProgress}%</span>
              </div>
              <Progress value={zipUploadProgress} className="h-2" />
            </div>
          )}

          {/* Success Message */}
          {successCount === files.length && files.length > 0 && (
            <div className="p-4 bg-green-500/10 rounded-lg border border-green-500/30">
              <p className="text-sm text-center text-green-400">
                ✅ All files uploaded successfully
              </p>
            </div>
          )}
        </TabsContent>

        {/* ========== URL/ZIP Download Tab ========== */}
        <TabsContent value="url" className="space-y-6">
          <div className="p-4 bg-purple-500/10 rounded-lg border border-purple-500/30">
            <p className="text-sm text-purple-300">
              📥 Download and extract datasets from URLs or file paths (supports HuggingFace, direct URLs, local paths)
            </p>
          </div>

          {/* Project Name */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Project Name
            </label>
            <input
              type="text"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              className={`w-full px-4 py-2 bg-slate-800 border rounded-lg text-white focus:ring-2 focus:border-transparent ${
                projectExists
                  ? 'border-yellow-500 focus:ring-yellow-500'
                  : 'border-slate-600 focus:ring-purple-500'
              }`}
              placeholder="my_awesome_character (no spaces or special chars)"
              disabled={downloading}
            />
            {projectExists ? (
              <p className="text-xs text-yellow-500 mt-1 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" />
                ⚠️ Dataset exists - downloaded files will be added to existing dataset
              </p>
            ) : (
              <p className="text-xs text-gray-500 mt-1">
                📁 Files will be downloaded to: datasets/{projectName}
              </p>
            )}
          </div>

          {/* Dataset URL */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Dataset URL or Path
            </label>
            <input
              type="text"
              value={datasetUrl}
              onChange={(e) => setDatasetUrl(e.target.value)}
              className="w-full px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              placeholder="/path/to/dataset.zip or https://huggingface.co/datasets/..."
              disabled={downloading}
            />
            <p className="text-xs text-gray-500 mt-1">
              💡 Supports: Local paths, HTTP(S) URLs, HuggingFace dataset URLs
            </p>
          </div>

          {/* Download Button */}
          <Button
            type="button"
            onClick={handleUrlDownload}
            disabled={downloading || !projectName.trim() || !datasetUrl.trim()}
            className="w-full gap-2 bg-gradient-to-r from-purple-500 to-purple-600 text-white hover:from-purple-600 hover:to-purple-700"
          >
            <Download className="w-5 h-5" />
            {downloading ? 'Downloading & Extracting...' : 'Download & Extract Dataset'}
          </Button>
        </TabsContent>

        {/* ========== Create Folder Tab ========== */}
        <TabsContent value="folder" className="space-y-6">
          <div className="p-4 bg-green-500/10 rounded-lg border border-green-500/30">
            <p className="text-sm text-green-300">
              📁 Create a new dataset folder with Kohya repeat format: <strong>{folderRepeats}_{folderName || 'folder_name'}</strong>
            </p>
          </div>

          {/* Folder Name */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Folder Name
            </label>
            <input
              type="text"
              value={folderName}
              onChange={(e) => setFolderName(e.target.value)}
              className={`w-full px-4 py-2 bg-slate-800 border rounded-lg text-white focus:ring-2 focus:border-transparent ${
                folderExists
                  ? 'border-red-500 focus:ring-red-500'
                  : 'border-slate-600 focus:ring-green-500'
              }`}
              placeholder="character_name"
              disabled={creatingFolder}
            />
            {folderExists && folderName.trim() && (
              <p className="text-xs text-red-500 mt-1 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" />
                ⚠️ Folder "{folderRepeats}_{folderName}" already exists - please choose a different name
              </p>
            )}
          </div>

          {/* Repeat Count */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Repeat Count
            </label>
            <input
              type="number"
              value={folderRepeats}
              onChange={(e) => setFolderRepeats(parseInt(e.target.value) || 1)}
              min={1}
              max={100}
              className="w-full px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-green-500 focus:border-transparent"
              disabled={creatingFolder}
            />
            <p className="text-xs text-gray-500 mt-1">
              💡 Kohya format: Higher repeats = more training emphasis on this dataset
            </p>
          </div>

          {/* Preview */}
          {folderName && (
            <div className="p-4 bg-slate-800 rounded-lg border border-slate-600">
              <p className="text-sm text-gray-400 mb-1">Final folder name:</p>
              <p className="text-lg font-bold text-green-400 font-mono">
                {folderRepeats}_{folderName}
              </p>
            </div>
          )}

          {/* Create Button */}
          <Button
            type="button"
            onClick={handleCreateFolder}
            disabled={creatingFolder || !folderName.trim()}
            className="w-full gap-2 bg-gradient-to-r from-green-500 to-green-600 text-white hover:from-green-600 hover:to-green-700"
          >
            <FolderPlus className="w-5 h-5" />
            {creatingFolder ? 'Creating...' : 'Create Folder'}
          </Button>
        </TabsContent>
      </Tabs>
    </div>
  );
}
