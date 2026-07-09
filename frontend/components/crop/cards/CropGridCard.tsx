'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Grid3x3, ChevronLeft, ChevronRight, X } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

import type { ImageWithTags } from '@/lib/api';
import { ASPECT_OPTIONS } from './CropSettingsCard';
import { Cropper, CropperImage, CropperArea } from '@/components/ui/cropper';

export interface ImageCropState {
  crop: { x: number; y: number };
  zoom: number;
  croppedAreaPixels: { x: number; y: number; width: number; height: number } | null;
  croppedPreview?: string;
}

interface CropGridCardProps {
  images: ImageWithTags[];
  aspectRatio: string;
  targetResolution: number;
  cropStates: Map<string, ImageCropState>;
  onCropChange: (filename: string, state: ImageCropState) => void;
}

export function CropGridCard({
  images,
  aspectRatio,
  targetResolution,
  cropStates,
  onCropChange,
}: CropGridCardProps) {
  const aspect = useMemo(() => {
    const opt = ASPECT_OPTIONS.find((o) => o.value === aspectRatio);
    return opt ? opt.w / opt.h : 1;
  }, [aspectRatio]);

  const [page, setPage] = useState(0);
  const [pageSizeState, setPageSizeState] = useState(25);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const gridRef = useRef<HTMLDivElement>(null);

  const totalPages = Math.ceil(images.length / pageSizeState);
  // Clamp page to valid range — handles stale state when images array changes
  const safePage = Math.min(page, Math.max(0, totalPages - 1));
  const visibleImages = images.slice(safePage * pageSizeState, (safePage + 1) * pageSizeState);

  const goToPage = (newPage: number) => {
    setPage(newPage);
    gridRef.current?.scrollTo(0, 0);
  };

  // Get the selected image data for the dialog
  const selectedImg = selectedImage
    ? images.find((i) => i.image_name === selectedImage)
    : null;

  // Derive a larger thumbnail URL for the dialog cropper (grid uses 192px, dialog uses 768px)
  const dialogSrc = selectedImg
    ? (selectedImg.url?.replace('size=192', 'size=768') ??
       `/api/dataset/serve/${selectedImg.image_name}/${selectedImg.image_name}`)
    : '';

  // Refs to store crop/zoom values across re-renders without triggering syncing effects
  const initialCropRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const initialZoomRef = useRef(1);
  const latestCropRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const latestZoomRef = useRef(1);
  const latestPixelCropRef = useRef<{ x: number; y: number; width: number; height: number } | null>(null);

  useEffect(() => {
    if (selectedImage) {
      const saved = cropStates.get(selectedImage);
      initialCropRef.current = saved?.crop ?? { x: 0, y: 0 };
      initialZoomRef.current = saved?.zoom ?? 1;
      latestCropRef.current = saved?.crop ?? { x: 0, y: 0 };
      latestZoomRef.current = saved?.zoom ?? 1;
      latestPixelCropRef.current = saved?.croppedAreaPixels ?? null;
    }
  }, [selectedImage, cropStates]);

  const handleSaveCrop = useCallback(async () => {
    if (!selectedImage || !selectedImg) return;

    const pixelCrop = latestPixelCropRef.current ?? cropStates.get(selectedImage)?.croppedAreaPixels ?? null;

    if (!pixelCrop) {
      // No crop defined — close without saving a preview
      setSelectedImage(null);
      return;
    }

    // Generate a cropped thumbnail preview using canvas
    try {
      // Load the source image (same URL the dialog uses)
      const img = new Image();
      // Use the same src resolution as the dialog (768px thumbnail)
      const src = selectedImg.url?.includes('size=192')
        ? selectedImg.url.replace('size=192', 'size=768')
        : `/api/dataset/serve/${selectedImg.image_name}/${selectedImg.image_name}`;

      const imageLoad = new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = reject;
      });
      img.src = src;
      await imageLoad;

      // Create a canvas at grid thumbnail size (192px)
      const canvas = document.createElement('canvas');
      canvas.width = 192;
      canvas.height = 192;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Canvas 2D context not available — privacy settings may be blocking it');

      // Draw the cropped area from the source image onto the 192x192 canvas
      // pixelCrop has {x, y, width, height} in the natural dimensions of the source image
      ctx.drawImage(
        img,
        pixelCrop.x, pixelCrop.y, pixelCrop.width, pixelCrop.height,
        0, 0, 192, 192,
      );

      const dataUrl = canvas.toDataURL('image/webp', 0.85);

      // Save state with the preview thumbnail
      onCropChange(selectedImage, {
        crop: latestCropRef.current,
        zoom: latestZoomRef.current,
        croppedAreaPixels: pixelCrop,
        croppedPreview: dataUrl,
      });
    } catch (err) {
      console.warn('Failed to generate crop preview, saving without preview:', err);
      // Save without preview as fallback
      onCropChange(selectedImage, {
        crop: latestCropRef.current,
        zoom: latestZoomRef.current,
        croppedAreaPixels: pixelCrop,
      });
    }

    setSelectedImage(null);
  }, [selectedImage, selectedImg, onCropChange, cropStates]);

  return (
    <Card className="border-violet-500/30 h-full flex flex-col">
      <CardHeader className="shrink-0">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Grid3x3 className="h-5 w-5 text-violet-400" />
              Image Grid
            </CardTitle>
            <CardDescription>
              Click an image to crop it — {images.length} images total
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary">
              {targetResolution}x{targetResolution}
            </Badge>
            <Badge variant="outline">{aspectRatio}</Badge>
            {totalPages > 1 && (
              <nav
                aria-label="Image grid pagination"
                className="flex items-center gap-2 ml-2 pl-2 border-l border-border"
              >
                <Select
                  value={String(pageSizeState)}
                  onValueChange={(v) => {
                    setPageSizeState(Number(v));
                    setPage(0);
                  }}
                >
                  <SelectTrigger className="h-8 w-[68px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="25">25</SelectItem>
                    <SelectItem value="50">50</SelectItem>
                    <SelectItem value="100">100</SelectItem>
                  </SelectContent>
                </Select>
                <Button
                  variant="outline"
                  size="icon"
                  className="h-8 w-8"
                  aria-label="Previous page"
                  disabled={safePage === 0}
                  onClick={() => goToPage(safePage - 1)}
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <span
                  className="text-sm text-muted-foreground whitespace-nowrap tabular-nums"
                  aria-live="polite"
                >
                  {safePage + 1} / {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="icon"
                  className="h-8 w-8"
                  aria-label="Next page"
                  disabled={safePage >= totalPages - 1}
                  onClick={() => goToPage(safePage + 1)}
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </nav>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 overflow-y-auto">
        <div
          ref={gridRef}
          className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 pb-4"
        >
          {visibleImages.map((img) => {
            const state = cropStates.get(img.image_name);
            const hasCrop = state?.croppedAreaPixels != null;
            const imgSrc =
              state?.croppedPreview ??
              img.url ??
              `/api/dataset/serve/${img.image_name}/${img.image_name}`;

            return (
              <div
                key={img.image_name}
                className="relative aspect-square rounded-md overflow-hidden border border-border/50 cursor-pointer hover:border-violet-500/50 transition-colors group"
                onClick={() => setSelectedImage(img.image_name)}
              >
                <img
                  src={imgSrc}
                  alt={img.image_name}
                  className="w-full h-full object-cover"
                  loading="lazy"
                />
                {/* Filename overlay on hover */}
                <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/60 to-transparent p-1.5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                  <p className="text-[10px] text-white truncate">
                    {img.image_name}
                  </p>
                </div>
                {/* Crop indicator badge */}
                {hasCrop && (
                  <div
                    className="absolute top-1.5 right-1.5 w-2.5 h-2.5 rounded-full bg-violet-400 shadow"
                    title="Cropped"
                  />
                )}
              </div>
            );
          })}
        </div>
        {totalPages > 1 && (
          <p className="text-center text-sm text-muted-foreground mt-2 pb-2">
            Showing{' '}
            {Math.min((safePage + 1) * pageSizeState, images.length)} of{' '}
            {images.length} images
          </p>
        )}
      </CardContent>

      {/* Crop overlay - simple portal-free overlay without Radix Dialog */}
      {selectedImage && selectedImg && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          onContextMenu={(e) => e.preventDefault()}
        >
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/80"
            onClick={() => setSelectedImage(null)}
          />
          {/* Content */}
          <div className="relative z-10 flex flex-col w-full max-w-4xl max-h-[90vh] bg-background border border-border rounded-lg shadow-2xl p-4 sm:p-6 gap-4 mx-4">
            {/* Header */}
            <div className="flex items-center gap-3 shrink-0">
              <h2 className="text-base font-semibold truncate flex-1">
                Crop: {selectedImage}
              </h2>
              <Badge variant="outline">{aspectRatio}</Badge>
              <Badge variant="secondary">{targetResolution}x{targetResolution}</Badge>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 shrink-0"
                onClick={() => setSelectedImage(null)}
                aria-label="Close crop"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
            {/* Cropper */}
            <div className="flex-1 min-h-0 flex rounded-md overflow-hidden bg-black/10 dark:bg-black/30">
              {selectedImg && (
                <Cropper
                  key={selectedImage}
                  crop={initialCropRef.current}
                  zoom={initialZoomRef.current}
                  aspectRatio={aspect}
                  allowOverflow={false}
                  onCropChange={(c) => {
                    latestCropRef.current = c;
                    latestPixelCropRef.current = null;
                    onCropChange(selectedImage, {
                      crop: c,
                      zoom: latestZoomRef.current,
                      croppedAreaPixels: null,
                    });
                  }}
                  onZoomChange={(z) => {
                    latestZoomRef.current = z;
                    onCropChange(selectedImage, {
                      crop: latestCropRef.current,
                      zoom: z,
                      croppedAreaPixels: latestPixelCropRef.current,
                    });
                  }}
                  onCropComplete={(_, pixelCrop) => {
                    latestPixelCropRef.current = pixelCrop;
                    onCropChange(selectedImage, {
                      crop: latestCropRef.current,
                      zoom: latestZoomRef.current,
                      croppedAreaPixels: pixelCrop,
                    });
                  }}
                  className="flex-1 min-h-0 w-full"
                >
                  <CropperImage
                    src={dialogSrc}
                    alt={selectedImage}
                    objectFit="cover"
                  />
                  <CropperArea shape="rectangle" withGrid />
                </Cropper>
              )}
            </div>
            {/* Save / Cancel */}
            <div className="flex items-center justify-end gap-2 shrink-0">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSelectedImage(null)}
              >
                Cancel
              </Button>
              <Button
                variant="default"
                size="sm"
                onClick={handleSaveCrop}
              >
                Save Crop
              </Button>
            </div>
            <p className="text-xs text-muted-foreground text-center shrink-0">
              Drag to position, scroll to zoom
            </p>
          </div>
        </div>
      )}
    </Card>
  );
}
