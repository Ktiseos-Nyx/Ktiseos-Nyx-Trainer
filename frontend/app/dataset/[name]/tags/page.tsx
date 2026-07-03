// app/dataset/[name]/tags/page.tsx
'use client';

import { useMemo, useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { toast } from 'sonner';
import { datasetAPI, ImageWithTags } from '@/lib/api';
import { Tag, Save, Loader2, CheckCircle, Home, Database, ImageIcon, Grid3x3, Grid2x2, LayoutGrid, Trash2, ArrowRightLeft, Crop } from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import {
  TagsInput,
  TagsInputList,
  TagsInputInput,
  TagsInputItem,
} from '@/components/ui/tags-input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { AspectRatio } from '@/components/ui/aspect-ratio';
import { TagViewer, BLANK_TAG } from '@/components/dataset/TagViewer';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';

type CardSize = 'small' | 'medium' | 'large';

const SIZE_CONFIG: Record<CardSize, { gridClasses: string; aspectRatio: number; maxTagsH: string }> = {
  small:  { gridClasses: 'gap-3 grid-cols-1 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5', aspectRatio: 1,     maxTagsH: 'max-h-16' },
  medium: { gridClasses: 'gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3',               aspectRatio: 4 / 5, maxTagsH: 'max-h-24' },
  large:  { gridClasses: 'gap-6 grid-cols-1 md:grid-cols-2',                               aspectRatio: 7 / 9, maxTagsH: 'max-h-32' },
};

/**
 * Renders the dataset tag editor page for the dataset specified in the current route.
 *
 * Presents a Tag Viewer with frequency chips (click to filter images), bulk
 * remove/replace via the Actions menu, per-image inline tag editors, card-size
 * controls, and a Save All button.
 *
 * @returns A React element rendering the full tag editor UI.
 */
export default function DatasetTagsPage() {
  const params = useParams();
  const datasetName = params.name as string;
  const [images, setImages] = useState<ImageWithTags[]>([]);
  const [loading, setLoading] = useState(true);
  const [localImages, setLocalImages] = useState<ImageWithTags[]>([]);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [cardSize, setCardSize] = useState<CardSize>('medium');
  const [deleting, setDeleting] = useState<string | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem('tag-editor-card-size');
    if (saved === 'small' || saved === 'medium' || saved === 'large') setCardSize(saved);
  }, []);

  const handleCardSizeChange = (size: CardSize) => {
    setCardSize(size);
    localStorage.setItem('tag-editor-card-size', size);
  };

  useEffect(() => {
    const loadImages = async () => {
      try {
        setLoading(true);
        const data = await datasetAPI.getImagesWithTags(datasetName);

        const seen = new Set<string>();
        const deduped = (data.images || [])
          .filter(img => {
            if (seen.has(img.image_path)) return false;
            seen.add(img.image_path);
            return true;
          })
          .map(img => ({
            ...img,
            tags: Array.from(new Set(img.tags || [])),
            has_tags: (img.tags || []).length > 0,
          }));

        setImages(deduped);
        setLocalImages(deduped.map(img => ({ ...img })));
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
    const uniqueTags = Array.from(new Set(newTags.filter(t => t.trim())));
    setLocalImages(prev =>
      prev.map(img =>
        img.image_path === imagePath
          ? { ...img, tags: uniqueTags, has_tags: uniqueTags.length > 0 }
          : img
      )
    );
  };

  /** Remove tags from all images locally — requires Save All to persist. */
  const handleRemoveTags = (tagsToRemove: string[]) => {
    const removeSet = new Set(tagsToRemove);
    setLocalImages(prev =>
      prev.map(img => {
        const tags = img.tags.filter(t => !removeSet.has(t));
        return { ...img, tags, has_tags: tags.length > 0 };
      })
    );
    toast.info('Tags removed. Click Save All to persist.');
  };

  /** Replace tags across all images locally — requires Save All to persist. */
  const handleReplaceTags = (replacements: Record<string, string>) => {
    setLocalImages(prev =>
      prev.map(img => {
        const tags = Array.from(
          new Set(img.tags.map(t => (replacements[t]?.trim() || t)).filter(t => t))
        );
        return { ...img, tags, has_tags: tags.length > 0 };
      })
    );
    toast.info('Tags replaced. Click Save All to persist.');
  };

  /** Filter the image grid by selected tags; no selection shows all images. */
  const filteredImages = useMemo(() => {
    if (selectedTags.length === 0) return localImages;
    return localImages.filter(img => {
      if (selectedTags.includes(BLANK_TAG) && img.tags.length === 0) return true;
      return selectedTags.some(t => t !== BLANK_TAG && img.tags.includes(t));
    });
  }, [selectedTags, localImages]);

  /** Delete an image and its caption from the dataset, then drop it from local and saved state. */
  const handleDeleteImage = async (imageName: string) => {
    setDeleting(imageName);
    try {
      await datasetAPI.deleteImage(datasetName, imageName);
      setLocalImages(prev => prev.filter(img => img.image_name !== imageName));
      setImages(prev => prev.filter(img => img.image_name !== imageName));
      toast.success(`Deleted ${imageName}`);
    } catch (err) {
      toast.error('Failed to delete image');
    } finally {
      setDeleting(null);
    }
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
      toast.error('Failed to save tags. Check console.');
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
            <p className="text-muted-foreground">
              {images.length} images
              {selectedTags.length > 0 && (
                <span className="ml-2 text-cyan-500">
                  · showing {filteredImages.length} filtered
                </span>
              )}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-card border border-border rounded-lg p-1">
              <Button variant={cardSize === 'large' ? 'default' : 'ghost'} size="sm" onClick={() => handleCardSizeChange('large')} className="h-8 px-3" title="Large cards">
                <Grid2x2 className="w-4 h-4" />
              </Button>
              <Button variant={cardSize === 'medium' ? 'default' : 'ghost'} size="sm" onClick={() => handleCardSizeChange('medium')} className="h-8 px-3" title="Medium cards">
                <Grid3x3 className="w-4 h-4" />
              </Button>
              <Button variant={cardSize === 'small' ? 'default' : 'ghost'} size="sm" onClick={() => handleCardSizeChange('small')} className="h-8 px-3" title="Small cards">
                <LayoutGrid className="w-4 h-4" />
              </Button>
            </div>
            <Button onClick={handleSaveAll} disabled={saving} className="bg-cyan-600 hover:bg-cyan-700">
              {saving ? <Loader2 className="w-5 h-5 animate-spin" /> : <Save className="w-5 h-5" />}
              {saved ? 'Saved!' : 'Save All'}
            </Button>
            <Link
              href={`/dataset/${datasetName}/convert`}
              className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-colors"
            >
              <ArrowRightLeft className="w-4 h-4" />
              Convert Format
            </Link>
            <Link
              href={`/dataset/${datasetName}/crop`}
              className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg font-semibold transition-colors"
            >
              <Crop className="w-4 h-4" />
              Crop
            </Link>
          </div>
        </div>

        {/* Tag Viewer */}
        <TagViewer
          images={localImages}
          selectedTags={selectedTags}
          onSelectedTagsChange={setSelectedTags}
          onRemoveTags={handleRemoveTags}
          onReplaceTags={handleReplaceTags}
        />

        {/* Image grid */}
        {filteredImages.length === 0 ? (
          <div className="text-center py-12">
            <ImageIcon className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
            <p className="text-muted-foreground">
              {selectedTags.length > 0
                ? 'No images match the selected tags.'
                : 'No images found in this dataset.'}
            </p>
            {selectedTags.length > 0 && (
              <Button variant="ghost" className="mt-3 text-cyan-500" onClick={() => setSelectedTags([])}>
                Clear filter
              </Button>
            )}
          </div>
        ) : (
          <div className={`grid ${SIZE_CONFIG[cardSize].gridClasses}`}>
            {filteredImages.map(img => (
              <div key={img.image_path} className="border rounded-lg overflow-hidden bg-card group">
                <AspectRatio ratio={SIZE_CONFIG[cardSize].aspectRatio} className="bg-muted">
                  <img
                    src={img.url || `/api/files/image/${datasetName}/${img.image_name}`}
                    alt={img.image_name}
                    className="h-full w-full rounded-t-lg object-contain"
                    loading="lazy"
                    decoding="async"
                    onError={e => (e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect width="200" height="200" fill="%23333"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%23999" font-family="sans-serif"%3ENo Image%3C/text%3E%3C/svg%3E')}
                  />
                </AspectRatio>
                <div className="p-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-xs text-muted-foreground truncate flex-1">
                      {img.image_name}
                    </div>
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 w-6 p-0 text-muted-foreground hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                          disabled={deleting === img.image_name}
                        >
                          {deleting === img.image_name ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                        </Button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>Delete image</AlertDialogTitle>
                          <AlertDialogDescription>
                            This will permanently delete <span className="font-mono">{img.image_name}</span> and its caption file.
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>Cancel</AlertDialogCancel>
                          <AlertDialogAction
                            className="bg-red-600 hover:bg-red-700"
                            onClick={() => handleDeleteImage(img.image_name)}
                          >
                            Delete
                          </AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
                  </div>
                  <TagsInput
                    value={img.tags}
                    onValueChange={newTags => updateTags(img.image_path, newTags)}
                    addOnPaste
                    className="w-full"
                  >
                    <TagsInputList className={`${SIZE_CONFIG[cardSize].maxTagsH} overflow-y-auto`}>
                      {img.tags.map((tag, index) => (
                        <TagsInputItem
                          key={`${img.image_path}-${tag}-${index}`}
                          value={tag}
                          className={selectedTags.includes(tag) ? 'border-cyan-500 text-cyan-400' : ''}
                        >
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
        )}
      </div>
    </div>
  );
}
