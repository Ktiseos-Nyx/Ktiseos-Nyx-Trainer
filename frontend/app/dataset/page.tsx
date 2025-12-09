'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Home, Database } from 'lucide-react';
import DatasetUploader from '@/components/DatasetUploader';
import Breadcrumbs from '@/components/Breadcrumbs';
import { datasetAPI, DatasetInfo } from '@/lib/api';
import { FolderOpen, Image as ImageIcon, Tag, Trash2, RefreshCw, Edit, Zap, Info } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

export default function DatasetPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
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
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Dataset Management', icon: <Database className="w-4 h-4" /> },
          ]}

        />

        {/* Page Header */}
        <div className="mb-8">
          <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
            <div>
              <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-green-400 via-teal-400 to-green-400 bg-clip-text text-transparent">
                Dataset Management
              </h1>
              <p className="text-xl text-muted-foreground">
                Upload and manage your training datasets
              </p>
            </div>

            {/* Quick Actions - Always visible */}
            {!loading && (
            <div className="flex flex-wrap gap-3">
              <Link
                href="/dataset/auto-tag"
                className={`px-4 py-2 ${datasets.length > 0
                  ? 'bg-gradient-to-r from-pink-500 to-rose-500 hover:from-pink-600 hover:to-rose-600'
                  : 'bg-muted cursor-not-allowed opacity-60'
                } text-white rounded-lg font-semibold transition-all shadow-lg flex items-center gap-2`}
                onClick={(e) => {
                  if (datasets.length === 0) {
                    e.preventDefault();
                    alert('📸 Upload a dataset first to use auto-tagging!');
                  }
                }}
              >
                <Zap className="w-4 h-4" />
                Auto-Tag
              </Link>
              <Link
                href="/dataset/tags"
                className={`px-4 py-2 ${datasets.length > 0
                  ? 'bg-gradient-to-r from-purple-500 to-indigo-500 hover:from-purple-600 hover:to-indigo-600'
                  : 'bg-muted cursor-not-allowed opacity-60'
                } text-white rounded-lg font-semibold transition-all shadow-lg flex items-center gap-2`}
                onClick={(e) => {
                  if (datasets.length === 0) {
                    e.preventDefault();
                    alert('🏷️ Upload a dataset first to edit tags!');
                  }
                }}
              >
                <Tag className="w-4 h-4" />
                Edit Tags
              </Link>
            </div>
            )}
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left: Uploader (2/3 width) */}
          <div className="lg:col-span-2">
            <DatasetUploader />
          </div>

          {/* Right: Dataset List (1/3 width) */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Existing Datasets</CardTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={loadDatasets}
                  className="gap-1"
                >
                  <RefreshCw className="w-4 h-4" />
                  Refresh
                </Button>
              </div>
            </CardHeader>
            <CardContent>

              {loading ? (
                <div className="text-center py-8 text-muted-foreground">
                  Loading...
                </div>
              ) : datasets.length === 0 ? (
                <div className="space-y-3">
                  {/* Placeholder cards - visual demo */}
                  {['Sample Dataset 1', 'Sample Dataset 2', 'Sample Dataset 3'].map((name, i) => (
                    <div
                      key={i}
                      className="border border-dashed border-border rounded-lg p-4 opacity-50 cursor-not-allowed"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <FolderOpen className="w-5 h-5 text-muted-foreground" />
                          <h3 className="font-semibold text-muted-foreground">{name}</h3>
                        </div>
                      </div>

                      <div className="text-sm text-muted-foreground space-y-1">
                        <div className="flex items-center gap-2">
                          <ImageIcon className="w-4 h-4" />
                          <span>0 images</span>
                        </div>

                        <div className="flex items-center gap-2">
                          <Tag className="w-4 h-4" />
                          <span className="text-yellow-600 dark:text-yellow-400">No tags</span>
                        </div>
                      </div>

                      <div className="mt-3 text-xs text-muted-foreground font-mono">
                        datasets/{name.toLowerCase().replace(' ', '_')}
                      </div>
                    </div>
                  ))}

                  <div className="text-center pt-4 pb-2">
                    <p className="text-sm text-muted-foreground">
                      👆 These are placeholder cards. Upload images above to create real datasets!
                    </p>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  {datasets.map((dataset) => (
                    <div
                      key={dataset.name}
                      className="border border-border rounded-lg p-4 hover:bg-accent/50 transition-colors"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <FolderOpen className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
                          <h3 className="font-semibold text-foreground">{dataset.name}</h3>
                        </div>
                        <button
                          onClick={() => handleDelete(dataset.name)}
                          className="text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 p-1"
                          title="Delete dataset"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>

                      <div className="text-sm text-muted-foreground space-y-1">
                        <div className="flex items-center gap-2">
                          <ImageIcon className="w-4 h-4" />
                          <span>{dataset.image_count} images</span>
                        </div>

                        <div className="flex items-center gap-2">
                          <Tag className="w-4 h-4" />
                          <span>
                            {dataset.tags_present ? (
                              <span className="text-green-600 dark:text-green-400">✓ Tagged</span>
                            ) : (
                              <span className="text-yellow-600 dark:text-yellow-400">No tags</span>
                            )}
                          </span>
                        </div>
                      </div>

                      <div className="mt-3 text-xs text-muted-foreground font-mono">
                        {dataset.path}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Workflow Guide */}
        <div className="mt-8">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Info className="w-5 h-5" />
                Dataset Workflow
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-lg font-semibold text-foreground">
                    <span className="flex h-8 w-8 items-center justify-center rounded-full bg-green-500/20 text-green-600 dark:text-green-400">1</span>
                    Upload Images
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Use the uploader above to add images to your dataset. Supports batch uploads and multiple formats.
                  </p>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-lg font-semibold text-foreground">
                    <span className="flex h-8 w-8 items-center justify-center rounded-full bg-pink-500/20 text-pink-600 dark:text-pink-400">2</span>
                    Auto-Tag
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Generate captions automatically using WD14 tagger models. Configure model, threshold, and batch size.
                  </p>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-lg font-semibold text-foreground">
                    <span className="flex h-8 w-8 items-center justify-center rounded-full bg-purple-500/20 text-purple-600 dark:text-purple-400">3</span>
                    Edit & Train
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Fine-tune tags, crop images, add trigger words. Then use your dataset for LoRA training.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
