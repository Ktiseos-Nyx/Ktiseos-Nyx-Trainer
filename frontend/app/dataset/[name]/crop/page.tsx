'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { useParams } from 'next/navigation';
import { Home, Database, Crop, Terminal, X, ChevronDown, ChevronUp } from 'lucide-react';
import { toast } from 'sonner';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Button } from '@/components/ui/button';
import {
  datasetAPI,
  datasetToolsAPI,
  type ImageWithTags,
  type CropJobStatus,
  type CropRegion,
} from '@/lib/api';
import { CropSettingsCard } from '@/components/crop/cards/CropSettingsCard';
import {
  CropGridCard,
  type ImageCropState,
} from '@/components/crop/cards/CropGridCard';
import { CropProgressCard } from '@/components/crop/cards/CropProgressCard';
import { Splitter, SplitterPanel } from '@/components/ui/splitter';
import { ScrollArea } from '@/components/ui/scroll-area';

const MAX_LOGS = 500;

/** Scale pixelCrop coordinates from 768px thumbnail space → original-image space.
 *  The thumbnail uses sharp's fit:'cover' centre-crop, so coordinates must be
 *  inverted from the scale+center-crop transform. */
function thumbToOriginal(
  pixelCrop: { x: number; y: number; width: number; height: number },
  origWidth: number,
  origHeight: number,
  thumbSize = 768,
): { x: number; y: number; width: number; height: number } {
  const scale = Math.max(thumbSize / origWidth, thumbSize / origHeight);
  const cropX = (origWidth * scale - thumbSize) / 2;
  const cropY = (origHeight * scale - thumbSize) / 2;
  return {
    x: Math.round((pixelCrop.x + cropX) / scale),
    y: Math.round((pixelCrop.y + cropY) / scale),
    width: Math.round(pixelCrop.width / scale),
    height: Math.round(pixelCrop.height / scale),
  };
}

export default function CropPage() {
  const params = useParams();
  const datasetName = params.name as string;

  // Images
  const [images, setImages] = useState<ImageWithTags[]>([]);
  const [imagesLoading, setImagesLoading] = useState(true);
  const [imageDims, setImageDims] = useState<Map<string, { width: number; height: number }>>(new Map());

  // Settings
  const [targetResolution, setTargetResolution] = useState(512);
  const [aspectRatio, setAspectRatio] = useState('1:1');
  const [outputMode, setOutputMode] = useState('new_dataset');
  const [outputFormat, setOutputFormat] = useState('webp');
  const [quality, setQuality] = useState(90);

  // Per-image crop states
  const [cropStates, setCropStates] = useState<Map<string, ImageCropState>>(new Map());

  // Job state
  const [isRunning, setIsRunning] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<CropJobStatus | null>(null);
  const [isComplete, setIsComplete] = useState(false);

  // Logs
  const [logs, setLogs] = useState<string[]>([]);
  const [logsExpanded, setLogsExpanded] = useState(true);

  const statusIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const lastFileRef = useRef<string | null>(null);

  // Cleanup
  useEffect(() => {
    return () => {
      if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);
    };
  }, []);

  // Load images
  useEffect(() => {
    if (!datasetName) return;
    setImagesLoading(true);
    datasetAPI
      .getImagesWithTags(datasetName)
      .then((res) => {
        const unique = res.images
          .filter(
            (img, idx, arr) => arr.findIndex((i) => i.image_path === img.image_path) === idx,
          )
          .map((img) => ({
            ...img,
            // Use thumbnail URL (192px for grid thumbnails; Dialog swaps to 768px)
            url: `/api/dataset-tools/thumbnail?path=${encodeURIComponent(img.image_path)}&size=192`,
          }));
        setImages(unique);

        // Batch-fetch original image dimensions from file headers (lightweight)
        Promise.allSettled(
          unique.map((img) =>
            fetch(
              `/api/dataset-tools/image-dims?path=${encodeURIComponent(img.image_path)}`,
            )
              .then((r) => (r.ok ? r.json() : null))
              .then(
                (d) =>
                  d
                    ? ([img.image_name, { width: d.width as number, height: d.height as number }] as const)
                    : null,
              ),
          ),
        ).then((results) => {
          const dims = new Map<string, { width: number; height: number }>();
          for (const r of results) {
            if (r.status === 'fulfilled' && r.value) {
              dims.set(r.value[0], r.value[1]);
            }
          }
          setImageDims(dims);
        });
      })
      .catch((err) => {
        console.error('Failed to load images:', err);
        toast.error('Failed to load dataset images');
      })
      .finally(() => setImagesLoading(false));
  }, [datasetName]);

  const addLog = useCallback((msg: string) => {
    setLogs((prev) => {
      const next = [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`];
      return next.length > MAX_LOGS ? next.slice(-MAX_LOGS) : next;
    });
  }, []);

  const stopPolling = useCallback(() => {
    if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);
  }, []);

  const startStatusPolling = useCallback(
    (id: string) => {
      if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);

      statusIntervalRef.current = setInterval(async () => {
        try {
          const status: CropJobStatus = await datasetToolsAPI.getCropStatus(id);
          setJobStatus(status);

          // Crop's FastAPI status carries no logs array, so synthesise a live
          // line per file from current_file. (The old /api/jobs/{id}/logs poll
          // 404'd — that job lives in FastAPI, not the Node job store.)
          if (status.current_file && status.current_file !== lastFileRef.current) {
            lastFileRef.current = status.current_file;
            addLog(`Cropping ${status.current_file} (${status.cropped_files}/${status.total_files})`);
          }

          if (
            status.status === 'completed' ||
            status.status === 'failed' ||
            status.status === 'cancelled'
          ) {
            stopPolling();
            setIsRunning(false);
            setIsComplete(true);

            if (status.status === 'completed' && status.result) {
              if (status.result.success) {
                toast.success(
                  `Cropped ${status.result.cropped}/${status.result.total} images → ${status.result.output_dir}`,
                );
              }
            } else if (status.status === 'failed') {
              toast.error('Crop failed');
            }
          }
        } catch {
          // Ignore polling errors
        }
      }, 1000);
    },
    [stopPolling, addLog],
  );

  const handleCropChange = useCallback(
    (filename: string, state: ImageCropState) => {
      setCropStates((prev) => {
        const next = new Map(prev);
        next.set(filename, state);
        return next;
      });
    },
    [],
  );

  const handleStart = async () => {
    if (!datasetName || images.length === 0) return;

    // Reset state
    setIsRunning(true);
    setIsComplete(false);
    setJobStatus(null);
    setLogs([]);
    lastFileRef.current = null;

    // Compute source regions for all images
    const crops: CropRegion[] = [];

    for (const img of images) {
      const state = cropStates.get(img.image_name);
      const dims = imageDims.get(img.image_name);

      if (state?.croppedAreaPixels) {
        const scaled = dims
          ? thumbToOriginal(state.croppedAreaPixels, dims.width, dims.height)
          : state.croppedAreaPixels;
        crops.push({
          filename: img.image_name,
          source_x: Math.round(scaled.x),
          source_y: Math.round(scaled.y),
          source_width: Math.round(scaled.width),
          source_height: Math.round(scaled.height),
        });
      } else {
        // No crop applied — use full image dimensions
        crops.push({
          filename: img.image_name,
          source_x: 0,
          source_y: 0,
          source_width: dims?.width ?? 512,
          source_height: dims?.height ?? 512,
        });
      }
    }

    addLog(`Prepared ${crops.length} crop regions`);

    try {
      const response = await datasetToolsAPI.cropImages({
        dataset_dir: datasetName,
        target_width: targetResolution,
        target_height: targetResolution,
        output_mode: outputMode as 'new_dataset' | 'in-place',
        output_format: outputFormat as 'webp' | 'jpg' | 'png',
        quality,
        crops,
      });

      if (response.success && response.job_id) {
        setJobId(response.job_id);
        addLog(`Started crop: ${response.message}`);
        startStatusPolling(response.job_id);
      } else {
        addLog(`❌ Failed: ${response.message}`);
        setIsRunning(false);
      }
    } catch (err) {
      addLog(`❌ Error: ${err}`);
      toast.error(`Crop failed: ${err}`);
      setIsRunning(false);
    }
  };

  const handleStop = async () => {
    if (!jobId) return;
    try {
      addLog('🛑 Stopping...');
      await datasetToolsAPI.stopCrop(jobId);
      addLog('✅ Stopped');
    } catch (err) {
      addLog(`⚠️ Stop request: ${err}`);
    } finally {
      setIsRunning(false);
      stopPolling();
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)]">
      {/* Header */}
      <div className="shrink-0 border-b border-border px-4 py-3">
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Dataset', href: '/dataset', icon: <Database className="w-4 h-4" /> },
            {
              label: datasetName,
              href: `/dataset/${datasetName}/tags`,
              icon: <Database className="w-4 h-4" />,
            },
            { label: 'Crop', icon: <Crop className="w-4 h-4" /> },
          ]}
        />
        <div className="mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-emerald-400 via-violet-400 to-emerald-400 bg-clip-text text-transparent">
            Batch Crop
          </h1>
          <p className="text-sm text-muted-foreground">
            Crop images in <span className="font-mono">{datasetName}</span> — drag to position, scroll to zoom
          </p>
        </div>
      </div>

      {imagesLoading ? (
        <div className="flex-1 flex items-center justify-center text-muted-foreground">Loading images...</div>
      ) : images.length === 0 ? (
        <div className="flex-1 flex items-center justify-center text-muted-foreground">No images found in dataset</div>
      ) : (
        <Splitter defaultSize={32} minSize={24} maxSize={55} className="flex-1 min-h-0">
          {/* Left: Settings + Progress + Logs */}
          <SplitterPanel>
            <div className="flex h-full flex-col">
              <ScrollArea className="flex-1 min-h-0">
                <div className="space-y-4 p-4">
                  <CropSettingsCard
                    targetResolution={targetResolution}
                    setTargetResolution={setTargetResolution}
                    aspectRatio={aspectRatio}
                    setAspectRatio={setAspectRatio}
                    outputMode={outputMode}
                    setOutputMode={setOutputMode}
                    outputFormat={outputFormat}
                    setOutputFormat={setOutputFormat}
                    quality={quality}
                    setQuality={setQuality}
                    disabled={isRunning}
                  />

                  <CropProgressCard
                    isRunning={isRunning}
                    status={jobStatus}
                    logs={logs}
                    onStart={handleStart}
                    onStop={handleStop}
                    canStart={images.length > 0}
                  />

                  {/* Logs */}
                  {logs.length > 0 && (
                    <div className="bg-card border border-border rounded-lg overflow-hidden">
                      <div className="flex items-center bg-accent/50 hover:bg-accent">
                        <Button
                          variant="ghost"
                          onClick={() => setLogsExpanded((v) => !v)}
                          aria-expanded={logsExpanded}
                          className="flex-1 px-4 py-3 justify-between h-auto"
                        >
                          <div className="flex items-center gap-2 font-semibold text-sm">
                            <Terminal className="w-4 h-4" />
                            Crop Logs{' '}
                            {jobId && (
                              <span className="text-xs text-muted-foreground">({jobId})</span>
                            )}
                          </div>
                          {logsExpanded ? (
                            <ChevronUp className="w-4 h-4" />
                          ) : (
                            <ChevronDown className="w-4 h-4" />
                          )}
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => setLogs([])}
                          className="mr-2"
                          aria-label="Clear logs"
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      </div>
                      {logsExpanded && (
                        <div className="p-3 bg-black/50 font-mono text-xs text-green-400 max-h-64 overflow-y-auto">
                          {logs.map((log, idx) => (
                            <div key={idx} className="py-0.5">
                              {log}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </ScrollArea>
            </div>
          </SplitterPanel>

          {/* Right: image grid (un-squished, fills the panel) */}
          <SplitterPanel>
            <div className="h-full p-4">
              <CropGridCard
                images={images}
                aspectRatio={aspectRatio}
                targetResolution={targetResolution}
                cropStates={cropStates}
                onCropChange={handleCropChange}
              />
            </div>
          </SplitterPanel>
        </Splitter>
      )}
    </div>
  );
}
