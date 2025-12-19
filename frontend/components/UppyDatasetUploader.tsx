'use client';

import { useState, useEffect } from 'react';
import Uppy from '@uppy/core';
import { Dashboard } from '@uppy/react';
import XHRUpload from '@uppy/xhr-upload';
import { API_BASE } from '@/lib/api';

// Uppy styles
import '@uppy/core/dist/style.css';
import '@uppy/dashboard/dist/style.css';
import '@/styles/uppy-custom.css';

export default function UppyDatasetUploader() {
  const [datasetName, setDatasetName] = useState('');

  // Initialize Uppy instance without event handlers
  const [uppy] = useState(() =>
    new Uppy({
      id: 'dataset-uploader',
      autoProceed: false,
      allowMultipleUploadBatches: true,
      restrictions: {
        // CHANGE 1: Bump to 10GB (10 * 1024 * 1024 * 1024)
        maxFileSize: 10 * 1024 * 1024 * 1024,
        allowedFileTypes: ['image/*', '.zip', '.tar', '.7z'], // Added tar/7z just in case
      },
    }).use(XHRUpload, {
      id: 'XHRUpload',
      endpoint: `${API_BASE}/dataset/upload-batch`,
      formData: true,
      fieldName: 'files',
      method: 'POST',
      limit: 3,
      timeout: 0, // Infinite timeout
    })
  );

  // Set up event handlers after uppy is initialized
  useEffect(() => {
    const handleUpload = () => {
      // Update endpoint with current dataset name before each upload
      const plugin = uppy.getPlugin('XHRUpload');
      if (plugin) {
        plugin.setOptions({
          endpoint: `${API_BASE}/dataset/upload-batch?dataset_name=${datasetName}`,
        });
      }

      // Validate dataset name
      if (!datasetName.trim()) {
        uppy.info('Please enter a dataset name!', 'error', 5000);
        uppy.cancelAll();
        return;
      }

      console.log(`📤 Starting upload to dataset: ${datasetName}`);
    };

    const handleUploadSuccess = (file: any, response: any) => {
      console.log('✅ Upload success:', file?.name);
    };

    const handleUploadError = (file: any, error: any, response: any) => {
      console.error('❌ Upload error:', file?.name, error);
    };

    const handleComplete = (result: any) => {
      console.log('🎉 Upload complete!', result);
      if (result.successful.length > 0) {
        uppy.info(
          `✅ Successfully uploaded ${result.successful.length} files!`,
          'success',
          5000
        );
      }
      if (result.failed.length > 0) {
        uppy.info(
          `❌ ${result.failed.length} files failed to upload`,
          'error',
          5000
        );
      }
    };

    // Register event handlers
    uppy.on('upload', handleUpload);
    uppy.on('upload-success', handleUploadSuccess);
    uppy.on('upload-error', handleUploadError);
    uppy.on('complete', handleComplete);

    // Cleanup handlers on unmount
    return () => {
      uppy.off('upload', handleUpload);
      uppy.off('upload-success', handleUploadSuccess);
      uppy.off('upload-error', handleUploadError);
      uppy.off('complete', handleComplete);
      (uppy as any).close({ reason: 'unmount' });
    };
  }, [uppy, datasetName]);

  return (
    <div className="container mx-auto p-8 max-w-6xl">
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
            Dataset Upload
          </h1>
          <p className="text-gray-400 mt-2">
            Upload images and ZIP files for your LoRA training dataset
          </p>
        </div>

        {/* Dataset Name Input */}
        <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-6">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Dataset Name
          </label>
          <input
            type="text"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
            placeholder="my_dataset"
            className="w-full px-4 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
          <p className="text-xs text-gray-500 mt-2">
            Files will be uploaded to: <code className="text-purple-400">dataset/{datasetName}/</code>
          </p>
        </div>

        {/* Uppy Dashboard */}
        <div className="uppy-wrapper">
          <Dashboard
            uppy={uppy}
            theme="dark"
            proudlyDisplayPoweredByUppy={false}
            width="100%"
            height={500}
            note="Images and ZIP files only, up to 500MB per file"
            metaFields={[
              { id: 'name', name: 'Name', placeholder: 'File name' },
            ]}
          />
        </div>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <h3 className="text-blue-400 font-semibold mb-2">📁 ZIP Files</h3>
            <p className="text-sm text-gray-400">
              Automatically extracted and flattened
            </p>
          </div>
          <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
            <h3 className="text-green-400 font-semibold mb-2">🔄 Smart Upload</h3>
            <p className="text-sm text-gray-400">
              Automatic retries and progress tracking
            </p>
          </div>
          <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
            <h3 className="text-purple-400 font-semibold mb-2">⚡ Fast Batches</h3>
            <p className="text-sm text-gray-400">
              Uploads 10 files at a time for speed
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
