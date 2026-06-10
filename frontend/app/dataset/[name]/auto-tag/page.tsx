'use client';

import { useState, useEffect, useRef } from 'react';
import { Home, Database, Tag, Terminal, X, ChevronDown, ChevronUp } from 'lucide-react';
import { toast } from 'sonner';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Form } from '@/components/ui/form';
import { datasetAPI, captioningAPI, BLIPConfig, GITConfig, LogPoller } from '@/lib/api';
import type { TaggingConfig } from '@/lib/api';
import { useTaggingForm } from '@/hooks/useTaggingForm';
import { getModelType } from '@/components/tagging/models';
import { BasicSettingsCard } from '@/components/tagging/cards/BasicSettingsCard';
import { ModelSettingsCard } from '@/components/tagging/cards/ModelSettingsCard';
import { TagProcessingCard } from '@/components/tagging/cards/TagProcessingCard';
import { PerformanceCard } from '@/components/tagging/cards/PerformanceCard';
import { ActionsCard } from '@/components/tagging/cards/ActionsCard';

const MAX_LOGS = 500;

export default function AutoTagPage() {
  const { form, datasets, loadingDatasets, reloadDatasets } = useTaggingForm();

  const [activeTab, setActiveTab] = useState('basics');

  // Job state
  const [tagging, setTagging] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [currentImage, setCurrentImage] = useState<string | null>(null);
  const [totalImages, setTotalImages] = useState<number | null>(null);

  // Logs — showLogs mounts the panel; logsExpanded controls body visibility
  const [showLogs, setShowLogs] = useState(true);
  const [logsExpanded, setLogsExpanded] = useState(true);
  const [logs, setLogs] = useState<string[]>([]);
  const logPollerRef = useRef<LogPoller | null>(null);
  const statusIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Auto-enable ONNX for v3 models (carried over from the original page).
  const selectedModel = form.watch('model');
  useEffect(() => {
    if (selectedModel.includes('-v3')) {
      form.setValue('useOnnx', true);
    }
  }, [selectedModel, form]);

  // Cleanup pollers on unmount
  useEffect(() => {
    return () => {
      if (logPollerRef.current) logPollerRef.current.stop();
      if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);
    };
  }, []);

  /** Append a timestamped message to the on-screen log, capping at MAX_LOGS entries. */
  const addLog = (msg: string) => {
    setLogs((prev) => {
      const next = [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`];
      return next.length > MAX_LOGS ? next.slice(-MAX_LOGS) : next;
    });
  };

  /** Append a raw (already-formatted) backend log line, capping at MAX_LOGS. */
  const appendRaw = (msg: string) => {
    setLogs((prev) => {
      const next = [...prev, msg];
      return next.length > MAX_LOGS ? next.slice(-MAX_LOGS) : next;
    });
  };

  const stopPolling = () => {
    if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);
    if (logPollerRef.current) logPollerRef.current.stop();
  };

  /** Poll the job status endpoint every second, updating progress and terminal states. */
  const startStatusPolling = (id: string) => {
    if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);
    statusIntervalRef.current = setInterval(async () => {
      try {
        const response = await datasetAPI.getTaggingStatus(id);
        const job = response.job ?? response;
        if (job.progress != null) setProgress(job.progress);
        if (job.current_image) setCurrentImage(job.current_image);
        if (job.total_images) setTotalImages(job.total_images);
        if (job.status === 'completed') {
          addLog('✅ Tagging completed successfully!');
          stopPolling();
          setTagging(false);
          setTimeout(reloadDatasets, 1000);
        } else if (job.status === 'failed') {
          addLog(`❌ Tagging failed: ${job.error}`);
          stopPolling();
          setTagging(false);
        } else if (job.status === 'cancelled') {
          addLog('🛑 Tagging was cancelled');
          stopPolling();
          setTagging(false);
        }
      } catch (err) {
        console.error('Status poll error:', err);
      }
    }, 1000);
  };

  /** HTTP log polling for WD14 jobs (WebSocket breaks through the Caddy proxy). */
  const connectWd14Logs = (id: string) => {
    if (logPollerRef.current) logPollerRef.current.stop();
    logPollerRef.current = datasetAPI.pollTaggingLogs(
      id,
      (data) => {
        if (data.type === 'log' && data.log) appendRaw(data.log);
        else if (data.type === 'progress' && data.progress != null) setProgress(data.progress as number);
        else if (data.type === 'status') {
          if (data.status === 'completed') addLog('✅ Tagging completed!');
          else if (data.status === 'failed') addLog('❌ Tagging failed');
        }
      },
      (error) => {
        console.error('Log polling error:', error);
        addLog('⚠️ Log polling error - status polling still active');
      },
    );
  };

  /** HTTP log polling for BLIP/GIT captioning jobs. */
  const connectCaptionLogs = (id: string) => {
    if (logPollerRef.current) logPollerRef.current.stop();
    logPollerRef.current = captioningAPI.pollLogs(
      id,
      (data) => {
        if (data.type === 'log' && data.log) appendRaw(data.log);
        else if (data.type === 'progress' && data.progress != null) setProgress(data.progress as number);
      },
      (error) => {
        console.error('Log polling error:', error);
        addLog('⚠️ Log polling error - status polling still active');
      },
    );
  };

  /** Validated submit — fired by RHF only when the config passes the zod schema. */
  const runTagging = async (cfg: TaggingConfig) => {
    const modelType = getModelType(cfg.model);
    setTagging(true);
    setShowLogs(true);
    setLogs([]);
    setProgress(0);
    setCurrentImage(null);
    setTotalImages(null);

    const methodLabel =
      modelType === 'wd14' ? 'WD14 Tagging' : modelType === 'blip' ? 'BLIP Captioning' : 'GIT Captioning';
    addLog(`🚀 Starting ${methodLabel}...`);
    addLog(`📁 Dataset: ${cfg.datasetDir}`);

    try {
      let response;
      if (modelType === 'blip') {
        const config: BLIPConfig = {
          dataset_dir: cfg.datasetDir,
          caption_extension: cfg.captionExtension,
          batch_size: cfg.batchSize,
          max_workers: cfg.maxWorkers,
          beam_search: cfg.blipBeamSearch,
          num_beams: cfg.blipNumBeams,
          top_p: cfg.blipTopP,
          max_length: cfg.blipMaxLength,
          min_length: cfg.blipMinLength,
          recursive: cfg.recursive,
          debug: cfg.debug,
        };
        response = await captioningAPI.startBLIP(config);
      } else if (modelType === 'git') {
        const config: GITConfig = {
          dataset_dir: cfg.datasetDir,
          caption_extension: cfg.captionExtension,
          model_id: cfg.model,
          batch_size: cfg.batchSize,
          max_workers: cfg.maxWorkers,
          max_length: cfg.gitMaxLength,
          remove_words: cfg.gitRemoveWords,
          recursive: cfg.recursive,
          debug: cfg.debug,
        };
        response = await captioningAPI.startGIT(config);
      } else {
        response = await datasetAPI.tag({
          datasetDir: cfg.datasetDir,
          model: cfg.model,
          forceDownload: cfg.forceDownload,
          threshold: cfg.threshold,
          generalThreshold: cfg.useGeneralThreshold ? cfg.generalThreshold : null,
          characterThreshold: cfg.useCharacterThreshold ? cfg.characterThreshold : null,
          captionExtension: cfg.captionExtension,
          captionSeparator: cfg.captionSeparator,
          undesiredTags: cfg.undesiredTags,
          tagReplacement: cfg.tagReplacement || null,
          alwaysFirstTags: cfg.alwaysFirstTags || null,
          characterTagsFirst: cfg.characterTagsFirst,
          useRatingTags: cfg.ratingTags !== 'none',
          useRatingTagsAsLastTag: cfg.ratingTags === 'last',
          removeUnderscore: cfg.removeUnderscore,
          characterTagExpand: cfg.characterTagExpand,
          overwriteMode: cfg.overwriteMode,
          recursive: cfg.recursive,
          batchSize: cfg.batchSize,
          maxWorkers: cfg.maxWorkers,
          useOnnx: cfg.useOnnx,
          frequencyTags: cfg.frequencyTags,
          debug: cfg.debug,
        });
      }

      if (response.success && response.job_id) {
        setJobId(response.job_id);
        addLog(`✅ Job started! ID: ${response.job_id}`);
        if (modelType === 'wd14') connectWd14Logs(response.job_id);
        else connectCaptionLogs(response.job_id);
        startStatusPolling(response.job_id);
      } else {
        addLog(`❌ Failed: ${response.message}`);
        setTagging(false);
      }
    } catch (err) {
      addLog(`❌ Error: ${err}`);
      toast.error(`${methodLabel} failed: ${err}`);
      setTagging(false);
    }
  };

  // Start runs zod validation first; on failure we surface the first error and
  // jump to Basics (where the required dataset/model live).
  const handleStart = form.handleSubmit(runTagging, (errors) => {
    const firstError = Object.values(errors)[0];
    const msg = (firstError?.message as string | undefined) ?? 'Please fix the highlighted fields';
    toast.warning(msg);
    setActiveTab('basics');
  });

  const handleStop = async () => {
    if (!jobId) return;
    try {
      addLog('🛑 Stopping...');
      await datasetAPI.stopTagging(jobId);
      addLog('✅ Stopped');
    } catch (err) {
      // Stop may fail if the process already exited — that's fine, still reset UI
      addLog(`⚠️ Stop request: ${err}`);
    } finally {
      setTagging(false);
      stopPolling();
    }
  };

  const canStart = !!form.watch('datasetDir') && datasets.length > 0;

  return (
    <div className="min-h-screen bg-background py-16">
      <div className="container mx-auto px-4 max-w-5xl">
        <Form {...form}>
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Dataset', href: '/dataset', icon: <Database className="w-4 h-4" /> },
            { label: 'Auto-Tagging', icon: <Tag className="w-4 h-4" /> },
          ]}
        />

        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-pink-400 via-rose-400 to-pink-400 bg-clip-text text-transparent">
            Auto-Tagging & Captioning
          </h1>
          <p className="text-xl text-muted-foreground mt-4">
            Generate tags or captions for your dataset using AI models
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="basics">Basics</TabsTrigger>
            <TabsTrigger value="model">Model</TabsTrigger>
            <TabsTrigger value="processing">Tag Processing</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
          </TabsList>

          <TabsContent value="basics" className="mt-6">
            <BasicSettingsCard form={form} datasets={datasets} loadingDatasets={loadingDatasets} />
          </TabsContent>
          <TabsContent value="model" className="mt-6">
            <ModelSettingsCard form={form} />
          </TabsContent>
          <TabsContent value="processing" className="mt-6">
            <TagProcessingCard form={form} />
          </TabsContent>
          <TabsContent value="performance" className="mt-6">
            <PerformanceCard form={form} />
          </TabsContent>
        </Tabs>

        {/* Always-visible action bar */}
        <div className="mt-6">
          <ActionsCard
            tagging={tagging}
            canStart={canStart}
            progress={progress}
            currentImage={currentImage}
            totalImages={totalImages}
            onStart={handleStart}
            onStop={handleStop}
          />
        </div>

        {/* Logs */}
        {showLogs && (
          <div className="mt-6">
            <div className="bg-card border border-border rounded-lg overflow-hidden">
              <div className="flex items-center bg-accent/50 hover:bg-accent">
                <Button
                  variant="ghost"
                  onClick={() => setLogsExpanded((v) => !v)}
                  aria-expanded={logsExpanded}
                  aria-controls="auto-tag-logs-body"
                  className="flex-1 px-6 py-4 justify-between h-auto"
                >
                  <div className="flex items-center gap-2 font-semibold">
                    <Terminal className="w-5 h-5" />
                    Process Logs {jobId && <span className="text-xs text-muted-foreground">({jobId})</span>}
                  </div>
                  {logsExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                </Button>
                {logs.length > 0 && (
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setLogs([])}
                    className="mr-2"
                    aria-label="Clear logs"
                  >
                    <X className="w-4 h-4" />
                  </Button>
                )}
              </div>
              {logsExpanded && (
                <div
                  id="auto-tag-logs-body"
                  className="p-4 bg-black/50 font-mono text-sm text-green-400 max-h-96 overflow-y-auto"
                >
                  {logs.length === 0 ? (
                    <div className="text-muted-foreground">No logs yet...</div>
                  ) : (
                    logs.map((log, idx) => (
                      <div key={idx} className="py-1">
                        {log}
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          </div>
        )}
          </Form>
      </div>
    </div>
  );
}
