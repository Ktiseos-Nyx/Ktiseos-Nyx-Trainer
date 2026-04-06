'use client';

import { useState, useEffect } from 'react';
import Uppy from '@uppy/core';
import { Dashboard } from '@uppy/react';
import XHRUpload from '@uppy/xhr-upload';
import { API_BASE } from '@/lib/api';
import { Input } from '@/components/ui/input';

// Uppy styles
import '@uppy/core/dist/style.css';
import '@uppy/dashboard/dist/style.css';
import '@/styles/uppy-custom.css';

const ARCHIVE_EXTENSIONS = ['.zip', '.tar', '.tar.gz', '.tgz', '.7z'];

function isArchive(filename: string): boolean {
  const lower = filename.toLowerCase();
  return ARCHIVE_EXTENSIONS.some(ext => lower.endsWith(ext));
}

/**
 * Dataset upload UI using Uppy. Accepts images and archive formats (ZIP, TAR, 7z).
 * Archives are routed to the upload-zip endpoint for server-side extraction;
 * images go to upload-batch. Up to 5 concurrent uploads.
 */
export default function UppyDatasetUploader() {
  const [datasetName, setDatasetName] = useState('');

  // Initialize Uppy instance without event handlers
  const [uppy] = useState(() =>
    new Uppy({
      id: 'dataset-uploader',
      autoProceed: false,
      allowMultipleUploadBatches: true,
      restrictions: {
        maxFileSize: 10 * 1024 * 1024 * 1024,
        // Use MIME types so compound extensions like .tar.gz work correctly
        allowedFileTypes: [
          'image/*',
          'application/zip',
          'application/x-zip-compressed',
          'application/gzip',
          'application/x-tar',
          'application/x-7z-compressed',
        ],
      },
    }).use(XHRUpload, {
      id: 'XHRUpload',
      endpoint: `${API_BASE}/dataset/upload-batch`,
      formData: true,
      fieldName: 'files',
      method: 'POST',
      limit: 5,
      timeout: 0, // Infinite timeout
    })
  );

  // Set up event handlers after uppy is initialized
  useEffect(() => {
    const handleUpload = async () => {
      // Validate dataset name
      if (!datasetName.trim()) {
        uppy.info('Please enter a dataset name!', 'error', 5000);
        uppy.cancelAll();
        return;
      }

      // Separate archives from images — archives use a different endpoint/contract
      const files = uppy.getFiles();
      const archiveFiles = files.filter(f => isArchive(f.name));

      // Upload archives in parallel via fetch
      // FastAPI expects field 'file' (singular) + 'dataset_name' form field
      if (archiveFiles.length > 0) {
        const results = await Promise.allSettled(
          archiveFiles.map(async (archiveFile) => {
            const formData = new FormData();
            formData.append('file', archiveFile.data as File);
            formData.append('dataset_name', datasetName);

            const res = await fetch(`${API_BASE}/dataset/upload-zip`, {
              method: 'POST',
              body: formData,
            });

            if (!res.ok) {
              const err = await res.text();
              throw new Error(err);
            }

            return archiveFile;
          })
        );

        let successCount = 0;
        for (const result of results) {
          if (result.status === 'fulfilled') {
            uppy.removeFile(result.value.id);
            successCount++;
          } else {
            console.error('❌ Archive upload failed:', result.reason);
          }
        }

        const failCount = results.length - successCount;
        if (failCount > 0) {
          uppy.info(`${failCount} archive(s) failed to upload`, 'error', 5000);
        }

        // If only archives were queued, show completion
        const remaining = uppy.getFiles();
        if (remaining.length === 0) {
          if (successCount > 0) {
            uppy.info(`✅ Uploaded ${successCount} archive(s)!`, 'success', 5000);
          }
          return;
        }
      }

      // Update endpoint for remaining image files
      const plugin = uppy.getPlugin('XHRUpload');
      if (plugin) {
        plugin.setOptions({
          endpoint: `${API_BASE}/dataset/upload-batch?dataset_name=${datasetName}`,
        });
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
          <Input
            type="text"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
            placeholder="my_dataset"
          />
          <p className="text-xs text-gray-500 mt-2">
            Files will be uploaded to: <code className="text-purple-400">datasets/{datasetName || '...'}/</code>
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
            note="Images, ZIP, TAR, TAR.GZ, and 7z files — up to 10GB per file"
            metaFields={[
              { id: 'name', name: 'Name', placeholder: 'File name' },
            ]}
          />
        </div>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <h3 className="text-blue-400 font-semibold mb-2">📁 Archives</h3>
            <p className="text-sm text-gray-400">
              ZIP, TAR, TAR.GZ, and 7z automatically extracted
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
              Uploads 5 files at a time for speed
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
