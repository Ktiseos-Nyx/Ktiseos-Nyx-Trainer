'use client';

import { useCallback, useMemo, useRef, useEffect } from 'react';
import { Grid3x3, RotateCcw } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ImageCropper, type ImageCropperCrop } from '@/components/ui/image-cropper';
import type { ImageWithTags } from '@/lib/api';
import { ASPECT_OPTIONS } from './CropSettingsCard';

export interface ImageCropState {
  crop: ImageCropperCrop;
  zoom: number;
  naturalWidth: number;
  naturalHeight: number;
}

interface CropGridCardProps {
  images: ImageWithTags[];
  aspectRatio: string;
  targetResolution: number;
  cropStates: Map<string, ImageCropState>;
  onCropChange: (filename: string, state: ImageCropState) => void;
  onFrameSize: (filename: string, w: number, h: number) => void;
  maxVisible?: number;
}

/**
 * Compute source pixel coordinates from crop state + frame size.
 *
 * The ImageCropper displays the image using "cover" sizing:
 *   scale = max(frameW / naturalW, frameH / naturalH) * zoom
 *
 * The crop position (x, y) is the image offset from frame center in screen pixels.
 */
export function computeSourceRegion(
  crop: ImageCropperCrop,
  zoom: number,
  frameW: number,
  frameH: number,
  naturalW: number,
  naturalH: number,
): { sx: number; sy: number; sw: number; sh: number } {
  if (frameW === 0 || frameH === 0 || naturalW === 0 || naturalH === 0) {
    return { sx: 0, sy: 0, sw: naturalW, sh: naturalH };
  }
  const scale = Math.max(frameW / naturalW, frameH / naturalH) * zoom;
  const sw = frameW / scale;
  const sh = frameH / scale;
  const sx = naturalW / 2 - crop.x / scale - sw / 2;
  const sy = naturalH / 2 - crop.y / scale - sh / 2;
  return { sx, sy, sw, sh };
}

/** Wrapper that measures the crop frame via ResizeObserver. */
function MeasureableCropFrame({
  filename,
  src,
  alt,
  aspect,
  crop,
  zoom,
  onCropChange,
  onZoomChange,
  onFrameSize,
}: {
  filename: string;
  src: string;
  alt: string;
  aspect: number;
  crop: ImageCropperCrop;
  zoom: number;
  onCropChange: (c: ImageCropperCrop) => void;
  onZoomChange: (z: number) => void;
  onFrameSize: (w: number, h: number) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect) onFrameSize(rect.width, rect.height);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [onFrameSize]);

  return (
    <div ref={containerRef} className="aspect-square">
      <ImageCropper
        src={src}
        alt={alt}
        aspect={aspect}
        shape="rect"
        grid
        crop={crop}
        zoom={zoom}
        onCropChange={onCropChange}
        onZoomChange={onZoomChange}
        className="h-full"
      />
    </div>
  );
}

export function CropGridCard({
  images,
  aspectRatio,
  targetResolution,
  cropStates,
  onCropChange,
  onFrameSize,
  maxVisible = 50,
}: CropGridCardProps) {
  const aspect = useMemo(() => {
    const opt = ASPECT_OPTIONS.find((o) => o.value === aspectRatio);
    return opt ? opt.w / opt.h : 1;
  }, [aspectRatio]);

  const visibleImages = images.slice(0, maxVisible);
  const hasMore = images.length > maxVisible;

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
              Drag to position, scroll to zoom — each image crops independently
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary">
              {targetResolution}x{targetResolution}
            </Badge>
            <Badge variant="outline">{aspectRatio}</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex-1 min-h-0">
        <ScrollArea className="h-full">
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 pr-4">
            {visibleImages.map((img) => {
              const state = cropStates.get(img.image_name);
              const imgSrc = img.url || `/api/dataset/serve/${img.image_name}/${img.image_name}`;

              return (
                <div key={img.image_name} className="space-y-1">
                  <div className="rounded-md overflow-hidden border border-border">
                    <MeasureableCropFrame
                      filename={img.image_name}
                      src={imgSrc}
                      alt={img.image_name}
                      aspect={aspect}
                      crop={state?.crop ?? { x: 0, y: 0 }}
                      zoom={state?.zoom ?? 1}
                      onCropChange={(c) =>
                        onCropChange(img.image_name, {
                          crop: c,
                          zoom: state?.zoom ?? 1,
                          naturalWidth: state?.naturalWidth ?? 0,
                          naturalHeight: state?.naturalHeight ?? 0,
                        })
                      }
                      onZoomChange={(z) =>
                        onCropChange(img.image_name, {
                          crop: state?.crop ?? { x: 0, y: 0 },
                          zoom: z,
                          naturalWidth: state?.naturalWidth ?? 0,
                          naturalHeight: state?.naturalHeight ?? 0,
                        })
                      }
                      onFrameSize={(w, h) => onFrameSize(img.image_name, w, h)}
                    />
                  </div>
                  <p
                    className="text-xs text-muted-foreground truncate px-0.5"
                    title={img.image_name}
                  >
                    {img.image_name}
                  </p>
                </div>
              );
            })}
          </div>
          {hasMore && (
            <p className="text-center text-sm text-muted-foreground mt-4">
              Showing {maxVisible} of {images.length} images.
            </p>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
