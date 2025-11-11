'use client';

import { useState, useEffect } from 'react';
import DatasetUploader from '@/components/DatasetUploader';
import { datasetAPI } from '@/lib/api';
import { FolderOpen, Image as ImageIcon, Tag, Trash2 } from 'lucide-react';

interface Dataset {
  name: string;
  path: string;
  image_count: number;
  tags_present: boolean;
}

export default function DatasetPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);

  // Load datasets
  const loadDatasets = async () => {
    try {
      setLoading(true);
      const data = await datasetAPI.list();
      setDatasets(data.datasets || []);
    } catch (err) {
      console.error('Failed to load datasets:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  // Delete dataset
  const handleDelete = async (name: string) => {
    if (!confirm(`Delete dataset "${name}"? This cannot be undone!`)) {
      return;
    }

    try {
      await fetch(`/api/dataset/${name}`, { method: 'DELETE' });
      loadDatasets();
    } catch (err) {
      alert(`Failed to delete: ${err}`);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-teal-50 dark:from-gray-900 dark:to-gray-800 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Dataset Management
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Upload and manage your training datasets
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left: Uploader (2/3 width) */}
          <div className="lg:col-span-2">
            <DatasetUploader />
          </div>

          {/* Right: Dataset List (1/3 width) */}
          <div className="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold">Existing Datasets</h2>
              <button
                onClick={loadDatasets}
                className="text-sm text-blue-500 hover:text-blue-700"
              >
                Refresh
              </button>
            </div>

            {loading ? (
              <div className="text-center py-8 text-gray-500">
                Loading...
              </div>
            ) : datasets.length === 0 ? (
              <div className="text-center py-12">
                <FolderOpen className="w-16 h-16 mx-auto text-gray-300 mb-4" />
                <p className="text-gray-500 text-sm">
                  No datasets yet.
                  <br />
                  Upload some images to get started!
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {datasets.map((dataset) => (
                  <div
                    key={dataset.name}
                    className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <FolderOpen className="w-5 h-5 text-blue-500" />
                        <h3 className="font-semibold">{dataset.name}</h3>
                      </div>
                      <button
                        onClick={() => handleDelete(dataset.name)}
                        className="text-red-500 hover:text-red-700 p-1"
                        title="Delete dataset"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>

                    <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      <div className="flex items-center gap-2">
                        <ImageIcon className="w-4 h-4" />
                        <span>{dataset.image_count} images</span>
                      </div>

                      <div className="flex items-center gap-2">
                        <Tag className="w-4 h-4" />
                        <span>
                          {dataset.tags_present ? (
                            <span className="text-green-600">✓ Tagged</span>
                          ) : (
                            <span className="text-yellow-600">No tags</span>
                          )}
                        </span>
                      </div>
                    </div>

                    <div className="mt-3 text-xs text-gray-400 font-mono">
                      {dataset.path}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Info Cards */}
        <div className="mt-8 grid md:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="bg-blue-100 dark:bg-blue-900/30 p-3 rounded-lg">
                <ImageIcon className="w-6 h-6 text-blue-600" />
              </div>
              <h3 className="font-semibold">Supported Formats</h3>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              PNG, JPG, JPEG, WebP, BMP
              <br />
              Recommended: 512x512 or 1024x1024
            </p>
          </div>

          <div className="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="bg-purple-100 dark:bg-purple-900/30 p-3 rounded-lg">
                <Tag className="w-6 h-6 text-purple-600" />
              </div>
              <h3 className="font-semibold">Auto-Tagging</h3>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Uses WD14 tagger to automatically caption your images with descriptive tags
            </p>
          </div>

          <div className="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded-lg">
                <FolderOpen className="w-6 h-6 text-green-600" />
              </div>
              <h3 className="font-semibold">Organization</h3>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Keep datasets organized in separate folders for easy management and training
            </p>
          </div>
        </div>

        {/* Benefits Banner */}
        <div className="mt-8 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg shadow-lg p-8 text-white text-center">
          <h3 className="text-2xl font-bold mb-4">
            Say Goodbye to Jupyter Widget Hell! 🎉
          </h3>
          <div className="grid md:grid-cols-3 gap-6 text-sm">
            <div>
              <div className="text-3xl mb-2">✅</div>
              <strong>Drag & Drop Works!</strong>
              <p className="text-indigo-100 mt-1">
                Upload hundreds of images at once without errors
              </p>
            </div>
            <div>
              <div className="text-3xl mb-2">🚀</div>
              <strong>No Race Conditions!</strong>
              <p className="text-indigo-100 mt-1">
                Proper state management with React hooks
              </p>
            </div>
            <div>
              <div className="text-3xl mb-2">💪</div>
              <strong>Actually Reliable!</strong>
              <p className="text-indigo-100 mt-1">
                No more mysterious upload failures
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
