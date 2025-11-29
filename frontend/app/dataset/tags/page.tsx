'use client';

import { useState, useEffect } from 'react';
import { datasetAPI, ImageWithTags } from '@/lib/api';
import { Tag, Plus, Minus, Replace, Zap, Loader2, CheckCircle, Save, X, Home, Database, ImageIcon } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';

export default function DatasetTagsPage() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [images, setImages] = useState<ImageWithTags[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'editor' | 'bulk' | 'trigger' | 'preview'>('preview');

  // Load available datasets
  useEffect(() => {
    const loadDatasets = async () => {
      try {
        const data = await datasetAPI.list();
        setDatasets(data.datasets || []);
        if (data.datasets && data.datasets.length > 0) {
          setSelectedDataset(data.datasets[0].path);
        }
      } catch (err) {
        console.error('Failed to load datasets:', err);
      }
    };
    loadDatasets();
  }, []);

  // Load images with tags when dataset changes
  useEffect(() => {
    if (!selectedDataset) return;

    const loadImages = async () => {
      try {
        setLoading(true);
        const data = await datasetAPI.getImagesWithTags(selectedDataset);
        setImages(data.images || []);
      } catch (err) {
        console.error('Failed to load images:', err);
      } finally {
        setLoading(false);
      }
    };

    loadImages();
  }, [selectedDataset]);

  const refreshImages = async () => {
    if (!selectedDataset) return;
    try {
      setLoading(true);
      const data = await datasetAPI.getImagesWithTags(selectedDataset);
      setImages(data.images || []);
    } catch (err) {
      console.error('Failed to refresh:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Breadcrumbs */}
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Dataset Management', href: '/dataset', icon: <Database className="w-4 h-4" /> },
            { label: 'Tag Editor', icon: <Tag className="w-4 h-4" /> },
          ]}
        />

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Tag Editor
          </h1>
          <p className="text-xl text-muted-foreground mt-4">
            Edit tags, bulk operations, and trigger word injection
          </p>
        </div>

        {/* Dataset Selector */}
        <div className="mb-6 bg-card backdrop-blur-sm rounded-lg border border-border p-6">
          <label className="block text-sm font-medium text-foreground mb-2">Select Dataset</label>
          <div className="flex gap-4">
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="flex-1 px-4 py-2 bg-background border border-border rounded-lg text-foreground focus:ring-2 focus:ring-primary focus:border-transparent"
            >
              {datasets.map((dataset) => (
                <option key={dataset.path} value={dataset.path}>
                  {dataset.name} ({dataset.image_count} images)
                </option>
              ))}
            </select>
            <button
              onClick={refreshImages}
              disabled={loading}
              className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : 'Refresh'}
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="mb-6">
          <div className="border-b border-border">
            <nav className="-mb-px flex space-x-8">
              {[
                { id: 'preview', label: 'Tag Preview', icon: Tag },
                { id: 'editor', label: 'Tag Editor', icon: Save },
                { id: 'bulk', label: 'Bulk Operations', icon: Zap },
                { id: 'trigger', label: 'Trigger Word', icon: Plus }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 transition-colors ${
                    activeTab === tab.id
                      ? 'border-cyan-500 text-cyan-400'
                      : 'border-transparent text-muted-foreground hover:text-foreground hover:border-muted'
                  }`}
                >
                  <tab.icon className="w-5 h-5" />
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'preview' && (
          <TagPreviewTab images={images} loading={loading} />
        )}
        {activeTab === 'editor' && (
          <TagEditorTab images={images} loading={loading} onUpdate={refreshImages} />
        )}
        {activeTab === 'bulk' && (
          <BulkOperationsTab datasetPath={selectedDataset} onComplete={refreshImages} />
        )}
        {activeTab === 'trigger' && (
          <TriggerWordTab datasetPath={selectedDataset} onComplete={refreshImages} />
        )}
      </div>
    </div>
  );
}

// ========== Tag Preview Tab ==========

function TagPreviewTab({ images, loading }: { images: ImageWithTags[], loading: boolean }) {
  if (loading) {
    return (
      <div className="text-center py-12">
        <Loader2 className="w-12 h-12 mx-auto text-muted-foreground animate-spin mb-4" />
        <p className="text-muted-foreground">Loading tags...</p>
      </div>
    );
  }

  const taggedCount = images.filter(img => img.has_tags).length;
  const allTags = images.flatMap(img => img.tags);
  const uniqueTags = Array.from(new Set(allTags));

  return (
    <div className="space-y-6">
      {/* Stats */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-card backdrop-blur-sm rounded-lg border border-border p-6">
          <div className="text-sm text-muted-foreground mb-1">Total Images</div>
          <div className="text-3xl font-bold text-foreground">{images.length}</div>
        </div>
        <div className="bg-card backdrop-blur-sm rounded-lg border border-border p-6">
          <div className="text-sm text-muted-foreground mb-1">Tagged Images</div>
          <div className="text-3xl font-bold text-green-400">{taggedCount}</div>
        </div>
        <div className="bg-card backdrop-blur-sm rounded-lg border border-border p-6">
          <div className="text-sm text-muted-foreground mb-1">Unique Tags</div>
          <div className="text-3xl font-bold text-cyan-400">{uniqueTags.length}</div>
        </div>
      </div>

      {/* Tag Cloud */}
      <div className="bg-card backdrop-blur-sm rounded-lg border border-border p-6">
        <h3 className="text-xl font-bold text-foreground mb-4">All Tags ({uniqueTags.length})</h3>
        <div className="flex flex-wrap gap-2">
          {uniqueTags.slice(0, 100).map((tag, i) => (
            <span
              key={i}
              className="px-3 py-1 bg-primary/10 text-primary rounded-full text-sm border border-primary/20"
            >
              {tag}
            </span>
          ))}
          {uniqueTags.length > 100 && (
            <span className="px-3 py-1 bg-muted text-muted-foreground rounded-full text-sm">
              +{uniqueTags.length - 100} more
            </span>
          )}
        </div>
      </div>

      {/* Images with Tags */}
      <div className="bg-card backdrop-blur-sm rounded-lg border border-border p-6">
        <h3 className="text-xl font-bold text-foreground mb-4">Images & Tags</h3>
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {images.slice(0, 20).map((img) => (
            <div key={img.image_path} className="border border-border rounded-lg p-3">
              <div className="font-semibold text-sm mb-2">{img.image_name}</div>
              {img.has_tags ? (
                <div className="flex flex-wrap gap-1">
                  {img.tags.map((tag, i) => (
                    <span key={i} className="px-2 py-0.5 bg-muted text-xs rounded">
                      {tag}
                    </span>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-muted-foreground italic">No tags</div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ========== Tag Editor Tab ==========

function TagEditorTab({ images, loading, onUpdate }: { images: ImageWithTags[], loading: boolean, onUpdate: () => void }) {
  const [selectedImage, setSelectedImage] = useState<ImageWithTags | null>(null);
  const [editedTags, setEditedTags] = useState<string[]>([]);
  const [newTag, setNewTag] = useState('');
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);

  const handleSelectImage = (img: ImageWithTags) => {
    setSelectedImage(img);
    setEditedTags(img.tags);
    setSuccess(false);
    setDialogOpen(true);
  };

  const addTag = () => {
    if (newTag.trim() && !editedTags.includes(newTag.trim())) {
      setEditedTags([...editedTags, newTag.trim()]);
      setNewTag('');
    }
  };

  const removeTag = (tag: string) => {
    setEditedTags(editedTags.filter(t => t !== tag));
  };

  const handleSave = async () => {
    if (!selectedImage) return;

    try {
      setSaving(true);
      await datasetAPI.updateTags(selectedImage.image_path, editedTags);
      setSuccess(true);
      setTimeout(() => {
        setSuccess(false);
        setDialogOpen(false);
      }, 1500);
      onUpdate();
    } catch (err) {
      console.error('Failed to save:', err);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <Loader2 className="w-12 h-12 mx-auto text-muted-foreground animate-spin mb-4" />
        <p className="text-muted-foreground">Loading images...</p>
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className="text-center py-12">
        <ImageIcon className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
        <p className="text-muted-foreground">No images found in this dataset</p>
      </div>
    );
  }

  return (
    <>
      {/* Image Gallery Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
        {images.map((img) => (
          <button
            key={img.image_path}
            onClick={() => handleSelectImage(img)}
            className="group relative aspect-square rounded-lg overflow-hidden border-2 border-border hover:border-primary transition-all bg-card hover:shadow-lg"
          >
            {/* Placeholder for image - would be replaced with actual image */}
            <div className="w-full h-full bg-gradient-to-br from-muted to-muted-foreground/10 flex items-center justify-center">
              <ImageIcon className="w-12 h-12 text-muted-foreground/50 group-hover:text-primary/50 transition-colors" />
            </div>

            {/* Overlay with filename and tag count */}
            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
              <div className="absolute bottom-0 left-0 right-0 p-3">
                <div className="text-white text-xs font-medium truncate mb-1">
                  {img.image_name}
                </div>
                <div className="flex items-center gap-1 text-xs text-white/80">
                  <Tag className="w-3 h-3" />
                  {img.tags.length} tags
                </div>
              </div>
            </div>

            {/* Tag indicator badge */}
            {img.has_tags && (
              <div className="absolute top-2 right-2 bg-green-500 text-white rounded-full p-1">
                <CheckCircle className="w-4 h-4" />
              </div>
            )}
          </button>
        ))}
      </div>

      {/* Tag Editor Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Tag className="w-5 h-5" />
              Edit Tags: {selectedImage?.image_name}
            </DialogTitle>
          </DialogHeader>

          {selectedImage && (
            <div className="space-y-4 mt-4">
              {/* Add Tag */}
              <div className="flex gap-2">
                <input
                  type="text"
                  value={newTag}
                  onChange={(e) => setNewTag(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && addTag()}
                  placeholder="Add new tag..."
                  className="flex-1 px-3 py-2 bg-background border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
                />
                <button
                  onClick={addTag}
                  className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
                >
                  <Plus className="w-5 h-5" />
                </button>
              </div>

              {/* Current Tags */}
              <div className="border border-border rounded-lg p-4 min-h-48 bg-muted/20">
                <div className="text-sm font-medium mb-3 text-muted-foreground">
                  Tags ({editedTags.length})
                </div>
                <div className="flex flex-wrap gap-2">
                  {editedTags.length === 0 ? (
                    <div className="text-sm text-muted-foreground italic">No tags yet</div>
                  ) : (
                    editedTags.map((tag, i) => (
                      <div
                        key={i}
                        className="flex items-center gap-1 px-3 py-1 bg-primary/10 text-primary rounded-full border border-primary/20"
                      >
                        <span className="text-sm">{tag}</span>
                        <button
                          onClick={() => removeTag(tag)}
                          className="hover:text-destructive transition-colors"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Save Button */}
              <button
                onClick={handleSave}
                disabled={saving}
                className="w-full flex items-center justify-center gap-2 bg-primary text-primary-foreground px-6 py-3 rounded-lg font-semibold hover:bg-primary/90 disabled:opacity-50 transition-colors"
              >
                {saving ? <Loader2 className="w-5 h-5 animate-spin" /> : <Save className="w-5 h-5" />}
                {saving ? 'Saving...' : 'Save Tags'}
              </button>

              {success && (
                <div className="flex items-center gap-2 text-green-600 dark:text-green-400 justify-center">
                  <CheckCircle className="w-5 h-5" />
                  Saved successfully!
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}

// ========== Bulk Operations Tab ==========

function BulkOperationsTab({ datasetPath, onComplete }: { datasetPath: string, onComplete: () => void }) {
  const [operation, setOperation] = useState<'add' | 'remove' | 'replace'>('add');
  const [tags, setTags] = useState('');
  const [replaceWith, setReplaceWith] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleExecute = async () => {
    if (!tags.trim()) return;

    try {
      setLoading(true);
      setResult(null);

      const tagList = tags.split(',').map(t => t.trim()).filter(t => t);

      const res = await datasetAPI.bulkTagOperation(datasetPath, operation, tagList, replaceWith);
      setResult(res);
      onComplete();
    } catch (err: any) {
      setResult({ success: false, error: err.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-card rounded-lg shadow-lg p-6 max-w-2xl mx-auto">
      <h3 className="text-xl font-bold text-foreground mb-4">Bulk Tag Operations</h3>

      <div className="space-y-4">
        {/* Operation Type */}
        <div>
          <label className="block text-sm font-medium text-foreground mb-2">Operation</label>
          <div className="grid grid-cols-3 gap-2">
            {[
              { value: 'add', label: 'Add Tags', icon: Plus },
              { value: 'remove', label: 'Remove Tags', icon: Minus },
              { value: 'replace', label: 'Replace Tags', icon: Replace }
            ].map((op) => (
              <button
                key={op.value}
                onClick={() => setOperation(op.value as any)}
                className={`flex items-center justify-center gap-2 px-4 py-3 rounded-lg border-2 ${
                  operation === op.value
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border hover:border-primary/50'
                }`}
              >
                <op.icon className="w-4 h-4" />
                {op.label}
              </button>
            ))}
          </div>
        </div>

        {/* Tags Input */}
        <div>
          <label className="block text-sm font-medium text-foreground mb-2">
            Tags (comma-separated)
          </label>
          <input
            type="text"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="1girl, solo, smile"
            className="w-full px-3 py-2 bg-input border border-border rounded-lg focus:ring-2 focus:ring-primary text-foreground"
          />
        </div>

        {/* Replace With (only for replace operation) */}
        {operation === 'replace' && (
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Replace With (optional)
            </label>
            <input
              type="text"
              value={replaceWith}
              onChange={(e) => setReplaceWith(e.target.value)}
              placeholder="Leave empty to remove"
              className="w-full px-3 py-2 bg-input border border-border rounded-lg focus:ring-2 focus:ring-primary text-foreground"
            />
          </div>
        )}

        {/* Execute Button */}
        <button
          onClick={handleExecute}
          disabled={loading || !tags.trim()}
          className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-purple-500 to-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-purple-600 hover:to-indigo-700 disabled:opacity-50"
        >
          {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Zap className="w-5 h-5" />}
          Execute {operation.charAt(0).toUpperCase() + operation.slice(1)}
        </button>

        {/* Result */}
        {result && (
          <div className={`p-4 rounded-lg ${result.success ? 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-400 border border-green-200 dark:border-green-500/30' : 'bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-400 border border-red-200 dark:border-red-500/30'}`}>
            {result.success ? (
              <div>
                <CheckCircle className="w-5 h-5 inline mr-2" />
                Operation completed successfully!
              </div>
            ) : (
              <div>Error: {result.error}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ========== Trigger Word Tab ==========

function TriggerWordTab({ datasetPath, onComplete }: { datasetPath: string, onComplete: () => void }) {
  const [triggerWord, setTriggerWord] = useState('');
  const [position, setPosition] = useState<'start' | 'end'>('start');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleInject = async () => {
    if (!triggerWord.trim()) return;

    try {
      setLoading(true);
      setResult(null);

      const res = await datasetAPI.injectTriggerWord(datasetPath, triggerWord.trim(), position);
      setResult(res);
      onComplete();
    } catch (err: any) {
      setResult({ success: false, error: err.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-card rounded-lg shadow-lg p-6 max-w-2xl mx-auto">
      <h3 className="text-xl font-bold text-foreground mb-4">Inject Trigger Word</h3>

      <div className="space-y-4">
        {/* Trigger Word Input */}
        <div>
          <label className="block text-sm font-medium text-foreground mb-2">Trigger Word</label>
          <input
            type="text"
            value={triggerWord}
            onChange={(e) => setTriggerWord(e.target.value)}
            placeholder="mycharacter"
            className="w-full px-3 py-2 bg-input border border-border rounded-lg focus:ring-2 focus:ring-primary text-foreground"
          />
          <p className="text-xs text-muted-foreground mt-1">
            This word will help the model recognize your character/style
          </p>
        </div>

        {/* Position */}
        <div>
          <label className="block text-sm font-medium text-foreground mb-2">Position</label>
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => setPosition('start')}
              className={`px-4 py-3 rounded-lg border-2 ${
                position === 'start'
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-border hover:border-primary/50'
              }`}
            >
              Start of caption
            </button>
            <button
              onClick={() => setPosition('end')}
              className={`px-4 py-3 rounded-lg border-2 ${
                position === 'end'
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-border hover:border-primary/50'
              }`}
            >
              End of caption
            </button>
          </div>
        </div>

        {/* Info Box */}
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
          <p className="text-sm text-foreground">
            <strong>Example:</strong> If your trigger word is "mychar" and position is "start":
            <br />
            <span className="font-mono text-xs bg-muted px-2 py-1 rounded mt-2 inline-block">
              1girl, smile → mychar, 1girl, smile
            </span>
          </p>
        </div>

        {/* Inject Button */}
        <button
          onClick={handleInject}
          disabled={loading || !triggerWord.trim()}
          className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-green-600 hover:to-emerald-700 disabled:opacity-50"
        >
          {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Zap className="w-5 h-5" />}
          Inject Trigger Word
        </button>

        {/* Result */}
        {result && (
          <div className={`p-4 rounded-lg ${result.success ? 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-400 border border-green-200 dark:border-green-500/30' : 'bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-400 border border-red-200 dark:border-red-500/30'}`}>
            {result.success ? (
              <div>
                <CheckCircle className="w-5 h-5 inline mr-2" />
                Trigger word injected into {result.modified} files!
              </div>
            ) : (
              <div>Error: {result.error}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
