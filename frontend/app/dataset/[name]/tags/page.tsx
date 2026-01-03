// app/dataset/[name]/tags/page.tsx
'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { datasetAPI, ImageWithTags } from '@/lib/api';
import { Tag, Save, Loader2, CheckCircle, Home, Database, ImageIcon, Grid3x3, Grid2x2, LayoutGrid } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import {
  TagsInput,
  TagsInputList,
  TagsInputInput,
  TagsInputItem,
} from '@/components/ui/tags-input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';

export default function DatasetTagsPage() {
  const params = useParams();
  const datasetName = params.name as string;
  const [images, setImages] = useState<ImageWithTags[]>([]);
  const [loading, setLoading] = useState(true);
  const [localImages, setLocalImages] = useState<ImageWithTags[]>([]);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set());
  const [bulkOperation, setBulkOperation] = useState<'add' | 'remove' | 'replace'>('add');
  const [bulkReplaceWith, setBulkReplaceWith] = useState('');

  // Card size preference (stored in localStorage)
  const [cardSize, setCardSize] = useState<'small' | 'medium' | 'large'>('medium');

  // Load card size preference from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('tag-editor-card-size');
    if (saved && (saved === 'small' || saved === 'medium' || saved === 'large')) {
      setCardSize(saved);
    }
  }, []);

  // Save card size preference to localStorage
  const handleCardSizeChange = (size: 'small' | 'medium' | 'large') => {
    setCardSize(size);
    localStorage.setItem('tag-editor-card-size', size);
  };

  useEffect(() => {
    const loadImages = async () => {
      try {
        setLoading(true);
        const data = await datasetAPI.getImagesWithTags(datasetName);

        // Deduplicate images by image_path and ensure tags are unique
        const seen = new Set<string>();
        const imagesWithTags = (data.images || [])
          .filter(img => {
            if (seen.has(img.image_path)) {
              return false; // Skip duplicate
            }
            seen.add(img.image_path);
            return true;
          })
          .map(img => ({
            ...img,
            // Deduplicate tags within each image
            tags: Array.from(new Set(img.tags || [])),
            has_tags: img.tags && img.tags.length > 0
          }));

        setImages(imagesWithTags);
        setLocalImages(imagesWithTags.map(img => ({ ...img })));
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

  const updateTags = (imagePath: string, newTags: string[]) => {
    // Deduplicate tags to prevent duplicates
    const uniqueTags = Array.from(new Set(newTags.filter(tag => tag.trim())));
    setLocalImages(prev =>
      prev.map(img =>
        img.image_path === imagePath
          ? { ...img, tags: uniqueTags, has_tags: uniqueTags.length > 0 }
          : img
      )
    );
  };

  // Get all unique tags from dataset
  const getAllUniqueTags = (): { tag: string; count: number }[] => {
    const tagCounts = new Map<string, number>();

    localImages.forEach(img => {
      img.tags.forEach(tag => {
        tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
      });
    });

    return Array.from(tagCounts.entries())
      .map(([tag, count]) => ({ tag, count }))
      .sort((a, b) => b.count - a.count); // Sort by frequency
  };

  const toggleTagSelection = (tag: string) => {
    setSelectedTags(prev => {
      const newSet = new Set(prev);
      if (newSet.has(tag)) {
        newSet.delete(tag);
      } else {
        newSet.add(tag);
      }
      return newSet;
    });
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
        <div className="mb-6 flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
          <div>
            <h1 className="text-4xl font-bold text-foreground">Tag Editor: {datasetName}</h1>
            <p className="text-muted-foreground">{images.length} images</p>
          </div>
          <div className="flex items-center gap-3">
            {/* Card Size Selector */}
            <div className="flex items-center gap-2 bg-card border border-border rounded-lg p-1">
              <Button
                variant={cardSize === 'large' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => handleCardSizeChange('large')}
                className="h-8 px-3"
                title="Large cards (2 columns)"
              >
                <Grid2x2 className="w-4 h-4" />
              </Button>
              <Button
                variant={cardSize === 'medium' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => handleCardSizeChange('medium')}
                className="h-8 px-3"
                title="Medium cards (3 columns)"
              >
                <Grid3x3 className="w-4 h-4" />
              </Button>
              <Button
                variant={cardSize === 'small' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => handleCardSizeChange('small')}
                className="h-8 px-3"
                title="Small cards (4-5 columns)"
              >
                <LayoutGrid className="w-4 h-4" />
              </Button>
            </div>

            <Button
              onClick={handleSaveAll}
              disabled={saving}
              className="bg-cyan-600 hover:bg-cyan-700"
            >
              {saving ? <Loader2 className="w-5 h-5 animate-spin" /> : <Save className="w-5 h-5" />}
              {saved ? 'Saved!' : 'Save All'}
            </Button>
          </div>
        </div>

        {/* Tag Library - Select Tags for Bulk Operations */}
        <div className="mb-8 p-6 bg-card border border-border rounded-lg shadow-sm">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <Tag className="w-5 h-5 text-cyan-500" />
              Tag Library
            </h2>
            {selectedTags.size > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">
                  {selectedTags.size} selected
                </span>
                <Button
                  onClick={() => setSelectedTags(new Set())}
                  variant="ghost"
                  size="sm"
                  className="text-cyan-500 hover:text-cyan-400"
                >
                  Clear
                </Button>
              </div>
            )}
          </div>
          <p className="text-sm text-muted-foreground mb-4">
            Click tags to select them for bulk operations below
          </p>
          <div className="flex flex-wrap gap-2 max-h-64 overflow-y-auto p-2 bg-background/50 rounded-md">
            {getAllUniqueTags().map(({ tag, count }) => (
              <Badge
                key={tag}
                onClick={() => toggleTagSelection(tag)}
                variant={selectedTags.has(tag) ? "default" : "outline"}
                className={`
                  cursor-pointer transition-colors
                  ${selectedTags.has(tag)
                    ? 'bg-cyan-500/20 border-cyan-500 text-cyan-400 hover:bg-cyan-500/30'
                    : 'hover:bg-muted/80 hover:border-muted-foreground/30'
                  }
                `}
              >
                {tag}
                <span className="ml-1.5 text-xs opacity-70">({count})</span>
              </Badge>
            ))}
            {getAllUniqueTags().length === 0 && (
              <p className="text-sm text-muted-foreground italic">No tags in dataset yet</p>
            )}
          </div>
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
              <Select value={bulkOperation} onValueChange={(value: 'add' | 'remove' | 'replace') => setBulkOperation(value)}>
                <SelectTrigger className="w-full md:w-40">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="add">Add Activation Tag (Prepend)</SelectItem>
                  <SelectItem value="remove">Remove Tags</SelectItem>
                  <SelectItem value="replace">Replace Tags</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2 flex-1">
              <label className="text-sm font-medium">
                Tags {selectedTags.size > 0 && <span className="text-cyan-500">({selectedTags.size} selected from library)</span>}
              </label>
              <div className="flex gap-2">
                <Input
                  type="text"
                  id="bulk-tags-input"
                  value={Array.from(selectedTags).join(', ')}
                  onChange={(e) => {
                    const tags = e.target.value.split(',').map(t => t.trim()).filter(t => t);
                    setSelectedTags(new Set(tags));
                  }}
                  placeholder="Select from library above or type tags here"
                  className="flex-1"
                />
                {bulkOperation === 'replace' && (
                  <Input
                    type="text"
                    id="bulk-replace-input"
                    placeholder="Replace with..."
                    value={bulkReplaceWith}
                    onChange={(e) => setBulkReplaceWith(e.target.value)}
                    className="flex-1"
                  />
                )}
              </div>
            </div>

            <Button
              onClick={async () => {
                const operation = bulkOperation;
                const tags = Array.from(selectedTags);
                const replaceWith = bulkReplaceWith.trim();

                if (tags.length === 0) {
                  alert('Please select at least one tag from the library or type tags in the input');
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
              className="bg-purple-600 hover:bg-purple-700"
            >
              Apply to All
            </Button>
          </div>
        </div>

        {/* Images with inline tag editors */}
        <div className={`grid gap-6 ${
          cardSize === 'small'
            ? 'grid-cols-1 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5'
            : cardSize === 'large'
            ? 'grid-cols-1 md:grid-cols-2'
            : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'
        }`}>
          {localImages.map((img) => (
            <div key={img.image_path} className="border rounded-lg overflow-hidden bg-card">
              <div className="aspect-square bg-muted flex items-center justify-center overflow-hidden">
                <img
                  src={img.url || `/api/files/image/${datasetName}/${img.image_name}`}
                  alt={img.image_name}
                  className="w-full h-full object-cover"
                  onError={(e) => (e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect width="200" height="200" fill="%23333"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%23999" font-family="sans-serif"%3ENo Image%3C/text%3E%3C/svg%3E')}
                />
              </div>
              <div className="p-3">
                <div className="text-xs text-muted-foreground mb-2 truncate">
                  {img.image_name}
                </div>
                <TagsInput
                  value={img.tags}
                  onValueChange={(newTags) => updateTags(img.image_path, newTags)}
                  addOnPaste
                  addOnBlur
                  className="w-full"
                >
                  <TagsInputList>
                    {img.tags.map((tag, index) => (
                      <TagsInputItem key={`${img.image_path}-${tag}-${index}`} value={tag}>
                        {tag}
                      </TagsInputItem>
                    ))}
                    <TagsInputInput placeholder="Add tags..." className="text-sm" />
                  </TagsInputList>
                </TagsInput>
                {img.has_tags && (
                  <div className="mt-1 flex items-center text-green-500 text-xs">
                    <CheckCircle className="w-3 h-3 mr-1" />
                    {img.tags.length} {img.tags.length === 1 ? 'tag' : 'tags'}
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
