"use client";

import * as React from "react";

import { cn } from "@/lib/utils";
import { Slider } from "@/components/ui/slider";

export type ImageCropperCrop = { x: number; y: number };

export type ImageCropperRef = {
  getCroppedDataUrl: (opts?: {
    size?: number;
    type?: string;
    quality?: number;
  }) => string | null;
  reset: () => void;
};

type Size = { w: number; h: number };

type ImageCropperContextValue = {
  zoom: number;
  setZoom: (next: number) => void;
  minZoom: number;
  maxZoom: number;
  disabled?: boolean;
};

const ImageCropperContext =
  React.createContext<ImageCropperContextValue | null>(null);

function useImageCropper() {
  const ctx = React.useContext(ImageCropperContext);
  if (!ctx) {
    throw new Error(
      "ImageCropper compound parts must be used inside <ImageCropper>",
    );
  }
  return ctx;
}

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

function coverScale(frame: Size, natural: Size) {
  return Math.max(frame.w / natural.w, frame.h / natural.h);
}

function clampCrop(
  crop: ImageCropperCrop,
  zoom: number,
  frame: Size | null,
  natural: Size | null,
): ImageCropperCrop {
  if (!frame || !natural || frame.w === 0 || frame.h === 0) return crop;
  const scale = coverScale(frame, natural) * zoom;
  const maxX = Math.max(0, (natural.w * scale - frame.w) / 2);
  const maxY = Math.max(0, (natural.h * scale - frame.h) / 2);
  return { x: clamp(crop.x, -maxX, maxX), y: clamp(crop.y, -maxY, maxY) };
}

export type ImageCropperProps = Omit<
  React.ComponentProps<"div">,
  "ref" | "onWheel" | "onDoubleClick"
> & {
  src: string;
  alt?: string;
  aspect?: number;
  shape?: "rect" | "round";
  zoom?: number;
  defaultZoom?: number;
  onZoomChange?: (zoom: number) => void;
  maxZoom?: number;
  crop?: ImageCropperCrop;
  defaultCrop?: ImageCropperCrop;
  onCropChange?: (crop: ImageCropperCrop) => void;
  grid?: boolean;
  disabled?: boolean;
  crossOrigin?: "" | "anonymous" | "use-credentials";
  ref?: React.Ref<ImageCropperRef>;
};

function ImageCropper({
  src,
  alt = "",
  aspect = 1,
  shape = "rect",
  zoom: zoomProp,
  defaultZoom = 1,
  onZoomChange,
  maxZoom = 3,
  crop: cropProp,
  defaultCrop,
  onCropChange,
  grid = false,
  disabled,
  crossOrigin = "anonymous",
  ref,
  className,
  children,
  ...props
}: ImageCropperProps) {
  const frameRef = React.useRef<HTMLDivElement | null>(null);
  const imgRef = React.useRef<HTMLImageElement | null>(null);

  const [naturalSize, setNaturalSize] = React.useState<Size | null>(null);
  const [frameSize, setFrameSize] = React.useState<Size | null>(null);
  const [dragging, setDragging] = React.useState(false);

  const [internalZoom, setInternalZoom] = React.useState(() =>
    clamp(defaultZoom, 1, maxZoom),
  );
  const zoom = zoomProp ?? internalZoom;

  const [internalCrop, setInternalCrop] = React.useState<ImageCropperCrop>(
    () => defaultCrop ?? { x: 0, y: 0 },
  );
  const crop = cropProp ?? internalCrop;

  const setZoom = React.useCallback(
    (next: number) => {
      const clamped = clamp(next, 1, maxZoom);
      if (zoomProp === undefined) setInternalZoom(clamped);
      onZoomChange?.(clamped);
    },
    [zoomProp, onZoomChange, maxZoom],
  );

  const setCrop = React.useCallback(
    (next: ImageCropperCrop) => {
      if (cropProp === undefined) setInternalCrop(next);
      onCropChange?.(next);
    },
    [cropProp, onCropChange],
  );

  const zoomRef = React.useRef(zoom);
  zoomRef.current = zoom;
  const cropRef = React.useRef(crop);
  cropRef.current = crop;
  const frameSizeRef = React.useRef(frameSize);
  frameSizeRef.current = frameSize;
  const naturalSizeRef = React.useRef(naturalSize);
  naturalSizeRef.current = naturalSize;

  const panBy = React.useCallback(
    (dx: number, dy: number) => {
      const c = cropRef.current;
      setCrop(
        clampCrop(
          { x: c.x + dx, y: c.y + dy },
          zoomRef.current,
          frameSizeRef.current,
          naturalSizeRef.current,
        ),
      );
    },
    [setCrop],
  );

  const zoomAt = React.useCallback(
    (next: number, focal?: ImageCropperCrop) => {
      const z0 = zoomRef.current;
      const z1 = clamp(next, 1, maxZoom);
      if (z1 === z0) return;
      let c = cropRef.current;
      if (focal) {
        const ratio = z1 / z0;
        c = {
          x: focal.x - ratio * (focal.x - c.x),
          y: focal.y - ratio * (focal.y - c.y),
        };
      }
      setCrop(clampCrop(c, z1, frameSizeRef.current, naturalSizeRef.current));
      setZoom(z1);
    },
    [maxZoom, setCrop, setZoom],
  );

  const setZoomCentered = React.useCallback(
    (next: number) => zoomAt(next, { x: 0, y: 0 }),
    [zoomAt],
  );

  const reset = React.useCallback(() => {
    const z = clamp(defaultZoom, 1, maxZoom);
    setZoom(z);
    setCrop(
      clampCrop(
        defaultCrop ?? { x: 0, y: 0 },
        z,
        frameSizeRef.current,
        naturalSizeRef.current,
      ),
    );
  }, [defaultZoom, defaultCrop, maxZoom, setZoom, setCrop]);

  React.useEffect(() => {
    const el = frameRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect) setFrameSize({ w: rect.width, h: rect.height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  React.useEffect(() => {
    const clamped = clampCrop(crop, zoom, frameSize, naturalSize);
    if (clamped.x !== crop.x || clamped.y !== crop.y) setCrop(clamped);
  }, [crop, zoom, frameSize, naturalSize, setCrop]);

  const wheelHandlerRef = React.useRef<(e: WheelEvent) => void>(() => {});
  wheelHandlerRef.current = (e: WheelEvent) => {
    if (disabled) return;
    e.preventDefault();
    const el = frameRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const focal = {
      x: e.clientX - rect.left - rect.width / 2,
      y: e.clientY - rect.top - rect.height / 2,
    };
    const factor = Math.exp(-e.deltaY * 0.002);
    zoomAt(zoomRef.current * factor, focal);
  };

  React.useEffect(() => {
    const el = frameRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => wheelHandlerRef.current(e);
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const pointersRef = React.useRef(new Map<number, ImageCropperCrop>());
  const pinchRef = React.useRef<{ dist: number; zoom: number } | null>(null);

  const pinchDistance = () => {
    const pts = Array.from(pointersRef.current.values());
    if (pts.length < 2) return 0;
    return Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (disabled) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    pointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointersRef.current.size === 2) {
      pinchRef.current = { dist: pinchDistance(), zoom: zoomRef.current };
    }
    setDragging(true);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (disabled) return;
    const prev = pointersRef.current.get(e.pointerId);
    if (!prev) return;
    pointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointersRef.current.size === 1) {
      panBy(e.clientX - prev.x, e.clientY - prev.y);
      return;
    }
    const pinch = pinchRef.current;
    const el = frameRef.current;
    if (!pinch || !el || pinch.dist === 0) return;
    const pts = Array.from(pointersRef.current.values());
    const rect = el.getBoundingClientRect();
    const focal = {
      x: (pts[0].x + pts[1].x) / 2 - rect.left - rect.width / 2,
      y: (pts[0].y + pts[1].y) / 2 - rect.top - rect.height / 2,
    };
    zoomAt(pinch.zoom * (pinchDistance() / pinch.dist), focal);
  };

  const handlePointerEnd = (e: React.PointerEvent<HTMLDivElement>) => {
    pointersRef.current.delete(e.pointerId);
    if (pointersRef.current.size < 2) pinchRef.current = null;
    if (pointersRef.current.size === 0) setDragging(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return;
    const step = e.shiftKey ? 20 : 5;
    switch (e.key) {
      case "ArrowLeft":
        panBy(step, 0);
        break;
      case "ArrowRight":
        panBy(-step, 0);
        break;
      case "ArrowUp":
        panBy(0, step);
        break;
      case "ArrowDown":
        panBy(0, -step);
        break;
      case "+":
      case "=":
        setZoomCentered(zoomRef.current + 0.1);
        break;
      case "-":
      case "_":
        setZoomCentered(zoomRef.current - 0.1);
        break;
      default:
        return;
    }
    e.preventDefault();
  };

  React.useImperativeHandle(
    ref,
    () => ({
      getCroppedDataUrl: (opts) => {
        const img = imgRef.current;
        const frame = frameSizeRef.current;
        const natural = naturalSizeRef.current;
        if (!img || !img.complete || !frame || !natural) return null;
        const scale = coverScale(frame, natural) * zoomRef.current;
        const sw = frame.w / scale;
        const sh = frame.h / scale;
        const sx = natural.w / 2 - cropRef.current.x / scale - sw / 2;
        const sy = natural.h / 2 - cropRef.current.y / scale - sh / 2;
        const outW = Math.max(1, Math.round(opts?.size ?? 512));
        const outH = Math.max(1, Math.round(outW / aspect));
        const canvas = document.createElement("canvas");
        canvas.width = outW;
        canvas.height = outH;
        const ctx2d = canvas.getContext("2d");
        if (!ctx2d) return null;
        ctx2d.drawImage(img, sx, sy, sw, sh, 0, 0, outW, outH);
        try {
          return canvas.toDataURL(opts?.type ?? "image/png", opts?.quality);
        } catch {
          return null;
        }
      },
      reset,
    }),
    [aspect, reset],
  );

  const ready = naturalSize !== null && frameSize !== null;
  const baseW = ready ? naturalSize.w * coverScale(frameSize, naturalSize) : 0;
  const baseH = ready ? naturalSize.h * coverScale(frameSize, naturalSize) : 0;

  const ctx = React.useMemo<ImageCropperContextValue>(
    () => ({
      zoom,
      setZoom: setZoomCentered,
      minZoom: 1,
      maxZoom,
      disabled,
    }),
    [zoom, setZoomCentered, maxZoom, disabled],
  );

  return (
    <ImageCropperContext.Provider value={ctx}>
      <div
        data-slot="image-cropper"
        className={cn("grid w-full gap-3", className)}
        {...props}
      >
        <div
          ref={frameRef}
          role="application"
          aria-label={alt ? `Crop ${alt}` : "Image cropper"}
          tabIndex={disabled ? -1 : 0}
          aria-disabled={disabled || undefined}
          data-slot="image-cropper-frame"
          data-shape={shape}
          data-dragging={dragging || undefined}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerEnd}
          onPointerCancel={handlePointerEnd}
          onDoubleClick={() => {
            if (!disabled) reset();
          }}
          onKeyDown={handleKeyDown}
          style={{ aspectRatio: aspect }}
          className={cn(
            "relative w-full select-none touch-none overflow-hidden rounded-sm border border-border bg-muted outline-none",
            "focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
            disabled
              ? "cursor-not-allowed opacity-60"
              : dragging
                ? "cursor-grabbing"
                : "cursor-grab",
          )}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            ref={imgRef}
            src={src}
            alt={alt}
            crossOrigin={crossOrigin}
            draggable={false}
            data-slot="image-cropper-image"
            onLoad={(e) =>
              setNaturalSize({
                w: e.currentTarget.naturalWidth,
                h: e.currentTarget.naturalHeight,
              })
            }
            className={cn(
              "pointer-events-none max-w-none",
              ready ? "absolute left-1/2 top-1/2" : "size-full object-cover",
            )}
            style={
              ready
                ? {
                    width: baseW,
                    height: baseH,
                    transform: `translate(calc(-50% + ${crop.x}px), calc(-50% + ${crop.y}px)) scale(${zoom})`,
                  }
                : undefined
            }
          />
          {shape === "round" && (
            <div
              aria-hidden
              data-slot="image-cropper-mask"
              className="pointer-events-none absolute inset-0 rounded-full shadow-[0_0_0_9999px_rgba(0,0,0,0.55)] ring-1 ring-white/40 ring-inset"
            />
          )}
          {grid && dragging && (
            <div
              aria-hidden
              data-slot="image-cropper-grid"
              className="pointer-events-none absolute inset-0"
            >
              <div className="absolute inset-y-0 left-1/3 w-px bg-white/40" />
              <div className="absolute inset-y-0 left-2/3 w-px bg-white/40" />
              <div className="absolute inset-x-0 top-1/3 h-px bg-white/40" />
              <div className="absolute inset-x-0 top-2/3 h-px bg-white/40" />
            </div>
          )}
        </div>
        {children}
      </div>
    </ImageCropperContext.Provider>
  );
}

type ImageCropperZoomProps = Omit<
  React.ComponentProps<typeof Slider>,
  "value" | "defaultValue" | "min" | "max" | "onValueChange"
>;

function ImageCropperZoom({ className, ...props }: ImageCropperZoomProps) {
  const ctx = useImageCropper();
  return (
    <Slider
      aria-label="Zoom"
      min={ctx.minZoom}
      max={ctx.maxZoom}
      step={0.01}
      value={[ctx.zoom]}
      onValueChange={([v]) => ctx.setZoom(v)}
      disabled={ctx.disabled}
      data-slot="image-cropper-zoom"
      className={cn("w-full", className)}
      {...props}
    />
  );
}

export { ImageCropper, ImageCropperZoom };
