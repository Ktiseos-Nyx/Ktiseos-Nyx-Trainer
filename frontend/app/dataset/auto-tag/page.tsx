'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { datasetAPI, DatasetInfo } from '@/lib/api';
import { Home, Database, Zap, FolderOpen, Image as ImageIcon, Tag, Loader2, ArrowRight } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

export default function SelectDatasetForAutoTagPage() {
  const router = useRouter();
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
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

    loadDatasets();
  }, []);

  const handleSelectDataset = (datasetName: string) => {
    router.push(`/dataset/${datasetName}/auto-tag`);
  };

  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-5xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Dataset Management', href: '/dataset', icon: <Database className="w-4 h-4" /> },
            { label: 'Auto-Tag', icon: <Zap className="w-4 h-4" /> },
          ]}
        />

        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-pink-400 via-rose-400 to-pink-400 bg-clip-text text-transparent">
            Select Dataset to Auto-Tag
          </h1>
          <p className="text-xl text-muted-foreground">
            Choose a dataset to automatically generate tags using WD14 tagger models
          </p>
        </div>

        {/* Dataset Selection */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FolderOpen className="w-5 h-5" />
              Available Datasets
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex flex-col items-center justify-center py-12">
                <Loader2 className="w-12 h-12 animate-spin text-muted-foreground mb-4" />
                <p className="text-muted-foreground">Loading datasets...</p>
              </div>
            ) : datasets.length === 0 ? (
              <div className="text-center py-12">
                <FolderOpen className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                <p className="text-muted-foreground mb-4">
                  No datasets found.
                  <br />
                  Upload some images first!
                </p>
                <Link
                  href="/dataset"
                  className="inline-block px-6 py-3 bg-gradient-to-r from-green-500 to-teal-500 hover:from-green-600 hover:to-teal-600 text-white rounded-lg font-semibold transition-all shadow-lg"
                >
                  Go to Dataset Upload
                </Link>
              </div>
            ) : (
              <div className="grid md:grid-cols-2 gap-4">
                {datasets.map((dataset) => (
                  <button
                    key={dataset.name}
                    onClick={() => handleSelectDataset(dataset.name)}
                    className="border border-border rounded-lg p-6 hover:bg-accent/50 hover:border-pink-500/50 transition-all text-left group"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <FolderOpen className="w-6 h-6 text-pink-600 dark:text-pink-400" />
                        <h3 className="font-semibold text-lg text-foreground">{dataset.name}</h3>
                      </div>
                      <ArrowRight className="w-5 h-5 text-muted-foreground group-hover:text-pink-500 transition-colors" />
                    </div>

                    <div className="space-y-2 text-sm text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <ImageIcon className="w-4 h-4" />
                        <span>{dataset.image_count} images</span>
                      </div>

                      <div className="flex items-center gap-2">
                        <Tag className="w-4 h-4" />
                        <span>
                          {dataset.tags_present ? (
                            <span className="text-green-600 dark:text-green-400">✓ Has tags</span>
                          ) : (
                            <span className="text-yellow-600 dark:text-yellow-400">Ready to tag</span>
                          )}
                        </span>
                      </div>
                    </div>

                    <div className="mt-3 text-xs text-muted-foreground font-mono truncate">
                      {dataset.path}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Info Box */}
        <div className="mt-6 p-4 bg-accent/30 border border-border rounded-lg">
          <h3 className="font-semibold text-foreground mb-2 flex items-center gap-2">
            <Zap className="w-4 h-4 text-pink-500" />
            About Auto-Tagging
          </h3>
          <p className="text-sm text-muted-foreground">
            Auto-tagging uses WD14 tagger models to automatically generate descriptive tags for your images.
            You can configure the model, threshold, and batch size on the next page.
          </p>
        </div>
      </div>
    </div>
  );
}
