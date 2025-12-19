// app/dataset/[name]/tags/page.tsx
'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { datasetAPI, ImageWithTags } from '@/lib/api';
import { Tag, Save, Loader2, CheckCircle, Home, Database, ImageIcon } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';

export default function DatasetTagsPage() {
  const params = useParams();
  const datasetName = params.name as string;
  const [images, setImages] = useState<ImageWithTags[]>([]);
  const [loading, setLoading] = useState(true);
  const [localImages, setLocalImages] = useState<ImageWithTags[]>([]);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    const loadImages = async () => {
      try {
        setLoading(true);
        const data = await datasetAPI.getImagesWithTags(datasetName);
        setImages(data.images || []);
        setLocalImages((data.images || []).map(img => ({ ...img })));
      } catch (err) {
        console.error('Failed to load images:', err);
        setImages([]);
        setLocalImages([]);
      } finally {
        setLoading(false);
      }
    };

    loadImages();
  }, [datasetName]);

  const updateTags = (imagePath: string, newTags: string) => {
    const tags = newTags
      .split(',')
      .map(t => t.trim())
      .filter(t => t);

    setLocalImages(prev =>
      prev.map(img =>
        img.image_path === imagePath
          ? { ...img, tags, has_tags: tags.length > 0 }
          : img
      )
    );
  };

  const handleSaveAll = async () => {
    setSaving(true);
    setSaved(false);
    try {
      for (const img of localImages) {
        const original = images.find(i => i.image_path === img.image_path);
        if (JSON.stringify(original?.tags) !== JSON.stringify(img.tags)) {
          await datasetAPI.updateTags(img.image_path, img.tags);
        }
      }
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (err) {
      console.error('Save failed:', err);
      alert('Failed to save tags. Check console.');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background py-16 flex flex-col items-center justify-center">
        <Loader2 className="w-12 h-12 animate-spin text-muted-foreground mb-4" />
        <p className="text-muted-foreground">Loading tags for "{datasetName}"...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Dataset Management', href: '/dataset', icon: <Database className="w-4 h-4" /> },
            { label: datasetName, href: `/dataset/${datasetName}` },
            { label: 'Tags', icon: <Tag className="w-4 h-4" /> },
          ]}
        />

        {/* Header */}
        <div className="mb-6 flex justify-between items-center">
          <div>
            <h1 className="text-4xl font-bold text-foreground">Tag Editor: {datasetName}</h1>
            <p className="text-muted-foreground">{images.length} images</p>
          </div>
          <button
            onClick={handleSaveAll}
            disabled={saving}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg flex items-center gap-2"
          >
            {saving ? <Loader2 className="w-5 h-5 animate-spin" /> : <Save className="w-5 h-5" />}
            {saved ? 'Saved!' : 'Save All'}
          </button>
        </div>

        {/* Bulk Operations */}
        <div className="mb-8 p-6 bg-card border border-border rounded-lg shadow-sm">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Database className="w-5 h-5 text-purple-500" />
            Bulk Operations
          </h2>
          <div className="grid md:grid-cols-[auto_1fr_auto] gap-4 items-end">
            <div className="space-y-2">
              <label className="text-sm font-medium">Operation</label>
              <select
                className="w-full md:w-40 px-3 py-2 bg-input border border-border rounded-md"
                id="bulk-op-select"
              >
                <option value="add">Add Activation Tag (Prepend)</option>
                <option value="remove">Remove Tags</option>
                <option value="replace">Replace Tags</option>
              </select>
            </div>
            
            <div className="space-y-2 flex-1">
              <label className="text-sm font-medium">Tags (comma separated)</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  id="bulk-tags-input"
                  placeholder="e.g. 1girl, solo"
                  className="flex-1 px-3 py-2 bg-input border border-border rounded-md"
                />
                <input
                  type="text"
                  id="bulk-replace-input"
                  placeholder="Replace with..."
                  className="hidden flex-1 px-3 py-2 bg-input border border-border rounded-md"
                />
              </div>
            </div>

            <button
              onClick={async () => {
                const opSelect = document.getElementById('bulk-op-select') as HTMLSelectElement;
                const tagsInput = document.getElementById('bulk-tags-input') as HTMLInputElement;
                const replaceInput = document.getElementById('bulk-replace-input') as HTMLInputElement;
                
                const operation = opSelect.value as 'add' | 'remove' | 'replace';
                const tags = tagsInput.value.split(',').map(t => t.trim()).filter(t => t);
                const replaceWith = replaceInput.value.trim();

                if (tags.length === 0) {
                  alert('Please enter at least one tag');
                  return;
                }

                if (confirm(`Are you sure you want to ${operation} tags for ALL images? This cannot be undone.`)) {
                  try {
                    setLoading(true);
                    const result = await datasetAPI.bulkTagOperation(
                      datasetName, // Use path as name here, strictly speaking it depends on API
                      operation,
                      tags,
                      replaceWith
                    );
                    
                    if (result.success) {
                      alert(`Successfully modified ${result.modified_count} files!`);
                      // Reload
                      window.location.reload();
                    } else {
                      alert('Operation failed');
                    }
                  } catch (e) {
                    console.error(e);
                    alert('Error performing bulk operation');
                  } finally {
                    setLoading(false);
                  }
                }
              }}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium transition-colors"
            >
              Apply to All
            </button>
          </div>
          
          <script dangerouslySetInnerHTML={{__html: `
            document.getElementById('bulk-op-select').addEventListener('change', function(e) {
              const val = e.target.value;
              const replaceInput = document.getElementById('bulk-replace-input');
              if (val === 'replace') {
                replaceInput.classList.remove('hidden');
              } else {
                replaceInput.classList.add('hidden');
              }
            });
          `}} />
        </div>

        {/* Images with inline tag editors */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {localImages.map((img) => (
            <div key={img.image_path} className="border rounded-lg overflow-hidden bg-card">
              <div className="aspect-square bg-muted flex items-center justify-center p-2">
                <img
                  src={img.url || `/api/files/image/${datasetName}/${img.image_name}`}
                  alt={img.image_name}
                  className="w-full h-full object-contain"
                  onError={(e) => (e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect width="200" height="200" fill="%23333"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%23999" font-family="sans-serif"%3ENo Image%3C/text%3E%3C/svg%3E')}
                />
              </div>
              <div className="p-3">
                <div className="text-xs text-muted-foreground mb-2 truncate">
                  {img.image_name}
                </div>
                <textarea
                  value={img.tags.join(', ')}
                  onChange={(e) => updateTags(img.image_path, e.target.value)}
                  placeholder="Enter tags, comma separated..."
                  className="w-full text-sm p-2 border rounded bg-background font-mono"
                  rows={2}
                />
                {img.has_tags && (
                  <div className="mt-1 flex items-center text-green-500 text-xs">
                    <CheckCircle className="w-3 h-3 mr-1" />
                    Tagged
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {localImages.length === 0 && (
          <div className="text-center py-12">
            <ImageIcon className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
            <p className="text-muted-foreground">No images found in this dataset.</p>
          </div>
        )}
      </div>
    </div>
  );
}
