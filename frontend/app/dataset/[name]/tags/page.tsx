// app/dataset/[name]/tags/page.tsx
'use client';

import { useState, useEffect } from 'react';
import { datasetAPI, ImageWithTags } from '@/lib/api';
import { Tag, Save, Loader2, CheckCircle, Home, Database, ImageIcon } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';

export default function DatasetTagsPage({ params }: { params: { name: string } }) {
  const datasetName = params.name;
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

        {/* Images with inline tag editors */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {localImages.map((img) => (
            <div key={img.image_path} className="border rounded-lg overflow-hidden bg-card">
              <div className="aspect-square bg-muted flex items-center justify-center p-2">
                <img
                  src={`/api/files/image/${datasetName}/${img.image_name}`}
                  alt={img.image_name}
                  className="w-full h-full object-contain"
                  onError={(e) => (e.currentTarget.src = '/placeholder.png')}
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
