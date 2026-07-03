'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { useParams } from 'next/navigation';
import { Home, Database, Crop, Terminal, X, ChevronDown, ChevronUp } from 'lucide-react';
import { toast } from 'sonner';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Button } from '@/components/ui/button';
import {
  datasetAPI,
  type ImageWithTags,
  type CropJobStatus,
  type CropRegion,
  type LogPoller,
} from '@/lib/api';
import { CropSettingsCard, ASPECT_OPTIONS } from '@/components/crop/cards/CropSettingsCard';
import {
  CropGridCard,
  type ImageCropState,
  computeSourceRegion,
} from '@/components/crop/cards/CropGridCard';
import { CropProgressCard } from '@/components/crop/cards/CropProgressCard';

const MAX_LOGS = 500;

export default function CropPage() {
  const params = useParams();
  const datasetName = params.name as string;

  // Images
  const [images, setImages] = useState<ImageWithTags[]>([]);
  const [imagesLoading, setImagesLoading] = useState(true);

  // Settings
  const [targetResolution, setTargetResolution] = useState(512);
  const [aspectRatio, setAspectRatio] = useState('1:1');
  const [outputMode, setOutputMode] = useState('new_dataset');
  const [outputFormat, setOutputFormat] = useState('webp');
  const [quality, setQuality] = useState(90);

  // Per-image crop states
  const [cropStates, setCropStates] = useState<Map<string, ImageCropState>>(new Map());

  // Frame sizes (measured from DOM via ResizeObserver)
  const [frameSizes, setFrameSizes] = useState<Map<string, { w: number; h: number }>>(new Map());

  // Job state
  const [isRunning, setIsRunning] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<CropJobStatus | null>(null);
  const [isComplete, setIsComplete] = useState(false);

  // Logs
  const [logs, setLogs] = useState<string[]>([]);
  const [logsExpanded, setLogsExpanded] = useState(true);

  const logPollerRef = useRef<LogPoller | null>(null);
  const statusIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup
  useEffect(() => {
    return () => {
      if (logPollerRef.current) logPollerRef.current.stop();
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
        const unique = res.images.filter(
          (img, idx, arr) => arr.findIndex((i) => i.image_path === img.image_path) === idx,
        );
        setImages(unique);
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
    if (logPollerRef.current) logPollerRef.current.stop();
  }, []);

  const startStatusPolling = useCallback(
    (id: string) => {
      if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);

      statusIntervalRef.current = setInterval(async () => {
        try {
          const status: CropJobStatus = await datasetAPI.getCropStatus(id);
          setJobStatus(status);

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
    [stopPolling],
  );

  const connectLogs = useCallback(
    (id: string) => {
      if (logPollerRef.current) logPollerRef.current.stop();

      logPollerRef.current = datasetAPI.pollCropLogs(
        id,
        (data) => {
          if (data.type === 'log') {
            setLogs((prev) => {
              const next = [...prev, data.log];
              return next.length > MAX_LOGS ? next.slice(-MAX_LOGS) : next;
            });
          }
        },
        (error) => {
          addLog(`⚠️ Log connection error: ${error.message}`);
        },
      );
    },
    [addLog],
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

  const handleFrameSize = useCallback((filename: string, w: number, h: number) => {
    setFrameSizes((prev) => {
      const next = new Map(prev);
      next.set(filename, { w, h });
      return next;
    });
  }, []);

  const handleStart = async () => {
    if (!datasetName || images.length === 0) return;

    // Reset state
    setIsRunning(true);
    setIsComplete(false);
    setJobStatus(null);
    setLogs([]);

    // Compute source regions for all images
    const crops: CropRegion[] = [];

    for (const img of images.slice(0, 50)) {
      const state = cropStates.get(img.image_name);
      const frameSize = frameSizes.get(img.image_name);

      const crop = state?.crop ?? { x: 0, y: 0 };
      const zoom = state?.zoom ?? 1;
      const naturalW = state?.naturalWidth ?? 512;
      const naturalH = state?.naturalHeight ?? 512;

      // Use measured frame size or estimate from grid layout
      const frameW = frameSize?.w ?? 200;
      const aspect = ASPECT_OPTIONS.find((o) => o.value === aspectRatio);
      const aspectW = aspect?.w ?? 1;
      const aspectH = aspect?.h ?? 1;
      const frameH = frameSize?.h ?? frameW / (aspectW / aspectH);

      const region = computeSourceRegion(crop, zoom, frameW, frameH, naturalW, naturalH);

      crops.push({
        filename: img.image_name,
        source_x: Math.round(region.sx),
        source_y: Math.round(region.sy),
        source_width: Math.round(region.sw),
        source_height: Math.round(region.sh),
      });
    }

    addLog(`Prepared ${crops.length} crop regions`);

    try {
      const response = await datasetAPI.cropImages({
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
        connectLogs(response.job_id);
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
      await datasetAPI.stopCrop(jobId);
      addLog('✅ Stopped');
    } catch (err) {
      addLog(`⚠️ Stop request: ${err}`);
    } finally {
      setIsRunning(false);
      stopPolling();
    }
  };

  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-7xl">
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

        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-emerald-400 via-violet-400 to-emerald-400 bg-clip-text text-transparent">
            Batch Crop
          </h1>
          <p className="text-xl text-muted-foreground mt-4">
            Crop images in <span className="font-mono">{datasetName}</span> — drag to position, scroll
            to zoom
          </p>
        </div>

        {imagesLoading ? (
          <div className="text-center py-12 text-muted-foreground">Loading images...</div>
        ) : images.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">No images found in dataset</div>
        ) : (
          <>
            {/* Settings + Progress Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
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
            </div>

            {/* Image Grid */}
            <CropGridCard
              images={images}
              aspectRatio={aspectRatio}
              targetResolution={targetResolution}
              cropStates={cropStates}
              onCropChange={handleCropChange}
              onFrameSize={handleFrameSize}
            />

            {/* Logs */}
            {logs.length > 0 && (
              <div className="mt-6">
                <div className="bg-card border border-border rounded-lg overflow-hidden">
                  <div className="flex items-center bg-accent/50 hover:bg-accent">
                    <Button
                      variant="ghost"
                      onClick={() => setLogsExpanded((v) => !v)}
                      aria-expanded={logsExpanded}
                      className="flex-1 px-6 py-4 justify-between h-auto"
                    >
                      <div className="flex items-center gap-2 font-semibold">
                        <Terminal className="w-5 h-5" />
                        Crop Logs{' '}
                        {jobId && (
                          <span className="text-xs text-muted-foreground">({jobId})</span>
                        )}
                      </div>
                      {logsExpanded ? (
                        <ChevronUp className="w-5 h-5" />
                      ) : (
                        <ChevronDown className="w-5 h-5" />
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
                    <div className="p-4 bg-black/50 font-mono text-sm text-green-400 max-h-96 overflow-y-auto">
                      {logs.map((log, idx) => (
                        <div key={idx} className="py-1">
                          {log}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
