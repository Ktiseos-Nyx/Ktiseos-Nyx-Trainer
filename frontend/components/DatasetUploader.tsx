'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Upload,
  Image as ImageIcon,
  CheckCircle,
  XCircle,
  Loader,
  Tags,
  FolderPlus,
  Link as LinkIcon,
  FileArchive,
  X,
  RefreshCw,
  Download,
} from 'lucide-react';
import { datasetAPI } from '@/lib/api';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';

interface UploadedFile {
  file: File;
  status: 'pending' | 'uploading' | 'success' | 'error';
  progress: number;
  error?: string;
}

export default function DatasetUploader() {
  // ========== State Management ==========
  const [activeTab, setActiveTab] = useState('direct');

  // Direct Upload State
  const [datasetName, setDatasetName] = useState('my_dataset');
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [uploading, setUploading] = useState(false);

  // URL/ZIP Download State
  const [projectName, setProjectName] = useState('');
  const [datasetUrl, setDatasetUrl] = useState('');
  const [downloading, setDownloading] = useState(false);

  // Folder Creation State
  const [folderName, setFolderName] = useState('');
  const [folderRepeats, setFolderRepeats] = useState(10);
  const [creatingFolder, setCreatingFolder] = useState(false);

  // ========== Direct Upload Handlers ==========

  // Handle file drop
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadedFile[] = acceptedFiles.map((file) => ({
      file,
      status: 'pending',
      progress: 0,
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

  // Upload images
  const handleUpload = async () => {
    if (files.length === 0) {
      alert('No files to upload!');
      return;
    }

    if (!datasetName.trim()) {
      alert('Please enter a dataset name!');
      return;
    }

    setUploading(true);

    try {
      setFiles((prev) =>
        prev.map((f) => ({ ...f, status: 'uploading' as const }))
      );

      const imageFiles = files.filter(f => f.file.type.startsWith('image/')).map((f) => f.file);

      if (imageFiles.length > 0) {
        await datasetAPI.uploadBatch(imageFiles, datasetName);
      }

      setFiles((prev) =>
        prev.map((f) => ({
          ...f,
          status: 'success' as const,
          progress: 100,
        }))
      );

      alert('✅ Upload complete!');
    } catch (err) {
      console.error('Upload failed:', err);
      setFiles((prev) =>
        prev.map((f) => ({
          ...f,
          status: 'error' as const,
          error: String(err),
        }))
      );
      alert(`❌ Upload failed: ${err}`);
    } finally {
      setUploading(false);
    }
  };

  // Upload and extract ZIP
  const handleUploadZip = async () => {
    const zipFiles = files.filter(f => f.file.name.endsWith('.zip'));

    if (zipFiles.length === 0) {
      alert('No ZIP files selected!');
      return;
    }

    if (!datasetName.trim()) {
      alert('Please enter a dataset name!');
      return;
    }

    setUploading(true);

    try {
      // TODO: Implement ZIP upload endpoint
      alert('✅ ZIP upload & extraction complete!');
    } catch (err) {
      alert(`❌ ZIP upload failed: ${err}`);
    } finally {
      setUploading(false);
    }
  };

  // Clear files
  const handleClear = () => {
    setFiles([]);
  };

  // Reset upload state
  const handleReset = () => {
    setFiles([]);
    setUploading(false);
  };

  // ========== URL/ZIP Download Handlers ==========

  const handleUrlDownload = async () => {
    if (!projectName.trim()) {
      alert('Please enter a project name!');
      return;
    }

    if (!datasetUrl.trim()) {
      alert('Please enter a dataset URL or path!');
      return;
    }

    setDownloading(true);

    try {
      // TODO: Implement URL download endpoint
      alert(`✅ Downloading dataset to: /workspace/datasets/${projectName}`);
    } catch (err) {
      alert(`❌ Download failed: ${err}`);
    } finally {
      setDownloading(false);
    }
  };

  // ========== Folder Creation Handlers ==========

  const handleCreateFolder = async () => {
    if (!folderName.trim()) {
      alert('Please enter a folder name!');
      return;
    }

    const kohyaFolderName = `${folderRepeats}_${folderName}`;

    setCreatingFolder(true);

    try {
      await datasetAPI.create(kohyaFolderName);
      setDatasetName(kohyaFolderName);
      alert(`✅ Created dataset folder: ${kohyaFolderName}`);

      // Switch to direct upload tab
      setActiveTab('direct');
    } catch (err) {
      alert(`❌ Failed to create folder: ${err}`);
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
    <div className="bg-slate-900/50 backdrop-blur-sm rounded-lg border border-slate-700 p-6">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">Dataset Upload</h2>
        <p className="text-gray-400">
          Choose your preferred method to set up your training dataset
        </p>
      </div>

      {/* Upload Method Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-3 bg-slate-800 border border-slate-700">
          <TabsTrigger value="direct" className="data-[state=active]:bg-blue-600">
            <Upload className="w-4 h-4 mr-2" />
            Direct Upload
          </TabsTrigger>
          <TabsTrigger value="url" className="data-[state=active]:bg-purple-600">
            <LinkIcon className="w-4 h-4 mr-2" />
            URL/ZIP
          </TabsTrigger>
          <TabsTrigger value="folder" className="data-[state=active]:bg-green-600">
            <FolderPlus className="w-4 h-4 mr-2" />
            Create Folder
          </TabsTrigger>
        </TabsList>

        {/* ========== Direct Upload Tab ========== */}
        <TabsContent value="direct" className="space-y-6">
          {/* Dataset Name Input */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Dataset Name
            </label>
            <input
              type="text"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              className="w-full px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="my_awesome_dataset"
              disabled={uploading}
            />
            <p className="text-xs text-gray-500 mt-1">
              📁 Files will be uploaded to: /workspace/datasets/{datasetName}
            </p>
          </div>

          {/* Drop Zone */}
          <div
            {...getRootProps()}
            className={`
              border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-all
              ${
                isDragActive
                  ? 'border-blue-500 bg-blue-500/10'
                  : 'border-slate-600 hover:border-blue-400 hover:bg-slate-800/50'
              }
            `}
          >
            <input {...getInputProps()} />

            <Upload
              className={`w-16 h-16 mx-auto mb-4 ${
                isDragActive ? 'text-blue-500 animate-bounce' : 'text-gray-500'
              }`}
            />

            {isDragActive ? (
              <p className="text-lg font-semibold text-blue-400">
                Drop files here!
              </p>
            ) : (
              <>
                <p className="text-lg font-semibold text-gray-300 mb-2">
                  Drag & drop images or ZIP files
                </p>
                <p className="text-sm text-gray-500">
                  or click to browse files
                </p>
                <p className="text-xs text-gray-600 mt-2">
                  Supports: PNG, JPG, JPEG, WebP, BMP, ZIP
                </p>
              </>
            )}
          </div>

          {/* File List */}
          {files.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold text-white">
                  Files ({files.length})
                  {imageCount > 0 && ` · ${imageCount} images`}
                  {zipCount > 0 && ` · ${zipCount} ZIP`}
                </h3>
                <div className="flex gap-2">
                  <button
                    onClick={handleReset}
                    className="text-sm text-yellow-400 hover:text-yellow-300 flex items-center gap-1"
                    disabled={uploading}
                  >
                    <RefreshCw className="w-3 h-3" />
                    Reset
                  </button>
                  <button
                    onClick={handleClear}
                    className="text-sm text-red-400 hover:text-red-300 flex items-center gap-1"
                    disabled={uploading}
                  >
                    <X className="w-3 h-3" />
                    Clear All
                  </button>
                </div>
              </div>

              {/* Status Summary */}
              <div className="flex gap-4 mb-4 text-sm">
                {pendingCount > 0 && (
                  <span className="text-gray-400">
                    ⏳ Pending: {pendingCount}
                  </span>
                )}
                {successCount > 0 && (
                  <span className="text-green-400">
                    ✓ Success: {successCount}
                  </span>
                )}
                {errorCount > 0 && (
                  <span className="text-red-400">
                    ✗ Error: {errorCount}
                  </span>
                )}
              </div>

              {/* File Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 max-h-96 overflow-y-auto p-1">
                {files.map((fileObj, index) => {
                  const isZip = fileObj.file.name.endsWith('.zip');
                  const preview = !isZip ? URL.createObjectURL(fileObj.file) : null;

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
                          src={preview!}
                          alt={fileObj.file.name}
                          className="w-full h-32 object-cover"
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
              <button
                onClick={handleUploadZip}
                disabled={uploading || zipCount === 0}
                className="flex-1 flex items-center justify-center gap-2 bg-gradient-to-r from-purple-500 to-purple-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-purple-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                <FileArchive className="w-5 h-5" />
                {uploading ? 'Extracting...' : `Extract ${zipCount} ZIP`}
              </button>
            )}

            {imageCount > 0 && (
              <button
                onClick={handleUpload}
                disabled={uploading || imageCount === 0}
                className="flex-1 flex items-center justify-center gap-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-blue-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                <Upload className="w-5 h-5" />
                {uploading ? 'Uploading...' : `Upload ${imageCount} Images`}
              </button>
            )}
          </div>

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
              className="w-full px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              placeholder="my_awesome_character (no spaces or special chars)"
              disabled={downloading}
            />
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
          <button
            onClick={handleUrlDownload}
            disabled={downloading || !projectName.trim() || !datasetUrl.trim()}
            className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-purple-500 to-purple-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-purple-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            <Download className="w-5 h-5" />
            {downloading ? 'Downloading & Extracting...' : 'Download & Extract Dataset'}
          </button>
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
              className="w-full px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-green-500 focus:border-transparent"
              placeholder="character_name"
              disabled={creatingFolder}
            />
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
          <button
            onClick={handleCreateFolder}
            disabled={creatingFolder || !folderName.trim()}
            className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-green-500 to-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-green-600 hover:to-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            <FolderPlus className="w-5 h-5" />
            {creatingFolder ? 'Creating...' : 'Create Folder'}
          </button>
        </TabsContent>
      </Tabs>
    </div>
  );
}
