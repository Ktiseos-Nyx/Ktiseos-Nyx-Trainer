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
} from 'lucide-react';
import { datasetAPI } from '@/lib/api';

interface UploadedFile {
  file: File;
  status: 'pending' | 'uploading' | 'success' | 'error';
  progress: number;
  error?: string;
}

export default function DatasetUploader() {
  const [datasetName, setDatasetName] = useState('my_dataset');
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [tagging, setTagging] = useState(false);

  // Handle file drop
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadedFile[] = acceptedFiles
      .filter((file) => file.type.startsWith('image/'))
      .map((file) => ({
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
    },
    multiple: true,
  });

  // Upload files
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
      // Update all files to uploading status
      setFiles((prev) =>
        prev.map((f) => ({ ...f, status: 'uploading' as const }))
      );

      // Upload batch
      const fileList = files.map((f) => f.file);
      await datasetAPI.uploadBatch(fileList, datasetName);

      // Mark all as success
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

      // Mark all as error
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

  // Auto-tag images
  const handleAutoTag = async () => {
    if (!datasetName.trim()) {
      alert('Please upload images first!');
      return;
    }

    setTagging(true);

    try {
      await datasetAPI.tag(`/workspace/datasets/${datasetName}`);
      alert('✅ Auto-tagging started! Check the logs.');
    } catch (err) {
      alert(`❌ Tagging failed: ${err}`);
    } finally {
      setTagging(false);
    }
  };

  // Create new dataset
  const handleCreateDataset = async () => {
    const name = prompt('New dataset name:');
    if (!name) return;

    try {
      await datasetAPI.create(name);
      setDatasetName(name);
      alert(`✅ Created dataset: ${name}`);
    } catch (err) {
      alert(`❌ Failed to create dataset: ${err}`);
    }
  };

  // Clear files
  const handleClear = () => {
    setFiles([]);
  };

  // Get status summary
  const successCount = files.filter((f) => f.status === 'success').length;
  const errorCount = files.filter((f) => f.status === 'error').length;
  const pendingCount = files.filter((f) => f.status === 'pending').length;

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Dataset Upload</h2>
        <p className="text-gray-600 dark:text-gray-400">
          Drag & drop your training images (no more broken upload widgets! 🎉)
        </p>
      </div>

      {/* Dataset Name Input */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">
          Dataset Name
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            placeholder="my_awesome_dataset"
            disabled={uploading}
          />
          <button
            onClick={handleCreateDataset}
            className="flex items-center gap-2 px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600"
            disabled={uploading}
          >
            <FolderPlus className="w-4 h-4" />
            New
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-1">
          Files will be uploaded to: /workspace/datasets/{datasetName}
        </p>
      </div>

      {/* Drop Zone */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-all
          ${
            isDragActive
              ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
              : 'border-gray-300 dark:border-gray-600 hover:border-blue-400'
          }
        `}
      >
        <input {...getInputProps()} />

        <Upload
          className={`w-16 h-16 mx-auto mb-4 ${
            isDragActive ? 'text-blue-500 animate-bounce' : 'text-gray-400'
          }`}
        />

        {isDragActive ? (
          <p className="text-lg font-semibold text-blue-600">
            Drop images here!
          </p>
        ) : (
          <>
            <p className="text-lg font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Drag & drop images here
            </p>
            <p className="text-sm text-gray-500">
              or click to browse files
            </p>
            <p className="text-xs text-gray-400 mt-2">
              Supports: PNG, JPG, JPEG, WebP, BMP
            </p>
          </>
        )}
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold">
              Files ({files.length})
            </h3>
            <button
              onClick={handleClear}
              className="text-sm text-red-500 hover:text-red-700"
              disabled={uploading}
            >
              Clear All
            </button>
          </div>

          {/* Status Summary */}
          <div className="flex gap-4 mb-4 text-sm">
            {pendingCount > 0 && (
              <span className="text-gray-600">
                ⏳ Pending: {pendingCount}
              </span>
            )}
            {successCount > 0 && (
              <span className="text-green-600">
                ✓ Success: {successCount}
              </span>
            )}
            {errorCount > 0 && (
              <span className="text-red-600">
                ✗ Error: {errorCount}
              </span>
            )}
          </div>

          {/* File Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 max-h-96 overflow-y-auto">
            {files.map((fileObj, index) => {
              const preview = URL.createObjectURL(fileObj.file);

              return (
                <div
                  key={index}
                  className="relative group rounded-lg overflow-hidden border-2 border-gray-200 dark:border-gray-700"
                >
                  {/* Image Preview */}
                  <img
                    src={preview}
                    alt={fileObj.file.name}
                    className="w-full h-32 object-cover"
                  />

                  {/* Status Overlay */}
                  <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                    <span className="text-white text-xs px-2 py-1 bg-black/70 rounded">
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
      <div className="mt-6 flex gap-3">
        <button
          onClick={handleUpload}
          disabled={uploading || files.length === 0}
          className="flex-1 flex items-center justify-center gap-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-blue-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          <Upload className="w-5 h-5" />
          {uploading ? 'Uploading...' : `Upload ${files.length} Files`}
        </button>

        <button
          onClick={handleAutoTag}
          disabled={tagging || uploading}
          className="flex items-center gap-2 bg-purple-500 text-white px-6 py-3 rounded-lg font-semibold hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          <Tags className="w-5 h-5" />
          {tagging ? 'Tagging...' : 'Auto-Tag'}
        </button>
      </div>

      {/* Success Message */}
      {successCount === files.length && files.length > 0 && (
        <div className="mt-4 p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg">
          <p className="text-sm text-center">
            ✅ <strong>All files uploaded successfully!</strong>
            <br />
            🎉 No upload errors! No race conditions! Everything just works!
          </p>
        </div>
      )}
    </div>
  );
}
