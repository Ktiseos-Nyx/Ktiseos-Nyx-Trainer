'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams } from 'next/navigation';
import { Home, Database, ArrowRightLeft, Terminal, X, ChevronDown, ChevronUp } from 'lucide-react';
import { toast } from 'sonner';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Button } from '@/components/ui/button';
import { datasetAPI } from '@/lib/api';
import type { ConvertJobStatus } from '@/lib/api';
import { ConvertSettingsCard } from '@/components/convert/cards/ConvertSettingsCard';
import { ConvertProgressCard } from '@/components/convert/cards/ConvertProgressCard';

const MAX_LOGS = 500;

export default function ConvertPage() {
  const params = useParams();
  const datasetName = params.name as string;

  // Settings state
  const [targetFormat, setTargetFormat] = useState('webp');
  const [quality, setQuality] = useState(90);
  const [outputMode, setOutputMode] = useState('new_dataset');

  // Job state
  const [isRunning, setIsRunning] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [totalFiles, setTotalFiles] = useState(0);
  const [convertedFiles, setConvertedFiles] = useState(0);
  const [currentFile, setCurrentFile] = useState<string | null>(null);
  const [isComplete, setIsComplete] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [result, setResult] = useState<ConvertJobStatus['result']>(null);

  // Logs
  const [showLogs] = useState(true);
  const [logsExpanded, setLogsExpanded] = useState(true);
  const [logs, setLogs] = useState<string[]>([]);

  const statusIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);
    };
  }, []);

  const addLog = useCallback((msg: string) => {
    setLogs((prev) => {
      const next = [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`];
      return next.length > MAX_LOGS ? next.slice(-MAX_LOGS) : next;
    });
  }, []);

  const stopPolling = useCallback(() => {
    if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);
  }, []);

  const startStatusPolling = useCallback((id: string) => {
    if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);

    statusIntervalRef.current = setInterval(async () => {
      try {
        const status: ConvertJobStatus = await datasetAPI.getConvertStatus(id);
        setProgress(status.progress);
        setConvertedFiles(status.converted_files);
        setCurrentFile(status.current_file || null);

        // Logs come from the status poll (FastAPI) — the job lives there, not in
        // the Node job store the old /api/jobs/{id}/logs poll hit (which 404'd).
        if (status.logs && status.logs.length > 0) {
          setLogs(status.logs.slice(-MAX_LOGS));
        }

        if (status.status === 'completed' || status.status === 'failed' || status.status === 'cancelled') {
          stopPolling();
          setIsRunning(false);
          setIsComplete(true);

          if (status.status === 'completed' && status.result) {
            setResult(status.result);
            if (status.result.success) {
              toast.success(`Converted ${status.result.converted}/${status.result.total} images`);
            }
          } else if (status.status === 'failed') {
            setHasError(true);
            toast.error('Conversion failed');
          }
        }
      } catch {
        // Ignore polling errors — will retry
      }
    }, 1000);
  }, [stopPolling]);

  const handleStart = async () => {
    if (!datasetName) return;

    // Reset state
    setIsRunning(true);
    setIsComplete(false);
    setHasError(false);
    setResult(null);
    setProgress(0);
    setConvertedFiles(0);
    setCurrentFile(null);
    setLogs([]);

    try {
      const response = await datasetAPI.convertFormat({
        dataset_dir: datasetName,
        target_format: targetFormat as 'webp' | 'jpg' | 'png' | 'bmp',
        quality,
        output_mode: outputMode as 'new_dataset' | 'in-place',
      });

      if (response.success && response.job_id) {
        setJobId(response.job_id);
        setTotalFiles(response.total_files);
        addLog(`Started conversion: ${response.message}`);
        startStatusPolling(response.job_id);
      } else {
        addLog(`❌ Failed: ${response.message}`);
        setIsRunning(false);
      }
    } catch (err) {
      addLog(`❌ Error: ${err}`);
      toast.error(`Conversion failed: ${err}`);
      setIsRunning(false);
    }
  };

  const handleStop = async () => {
    if (!jobId) return;
    try {
      addLog('🛑 Stopping...');
      await datasetAPI.stopConvert(jobId);
      addLog('✅ Stopped');
    } catch (err) {
      addLog(`⚠️ Stop request: ${err}`);
    } finally {
      setIsRunning(false);
      stopPolling();
    }
  };

  return (
    <div className="min-h-screen py-16">
      <div className="container mx-auto px-4 max-w-5xl">
        <Breadcrumbs
          items={[
            { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
            { label: 'Dataset', href: '/dataset', icon: <Database className="w-4 h-4" /> },
            { label: datasetName, href: `/dataset/${datasetName}/tags`, icon: <Database className="w-4 h-4" /> },
            { label: 'Convert', icon: <ArrowRightLeft className="w-4 h-4" /> },
          ]}
        />

        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-400 bg-clip-text text-transparent">
            Convert Format
          </h1>
          <p className="text-xl text-muted-foreground mt-4">
            Convert images in <span className="font-mono">{datasetName}</span> to a different format
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Settings */}
          <ConvertSettingsCard
            targetFormat={targetFormat}
            setTargetFormat={setTargetFormat}
            quality={quality}
            setQuality={setQuality}
            outputMode={outputMode}
            setOutputMode={setOutputMode}
            disabled={isRunning}
          />

          {/* Progress */}
          <ConvertProgressCard
            isRunning={isRunning}
            progress={progress}
            totalFiles={totalFiles}
            convertedFiles={convertedFiles}
            currentFile={currentFile}
            isComplete={isComplete}
            hasError={hasError}
            result={result}
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
                  aria-controls="convert-logs-body"
                  className="flex-1 px-6 py-4 justify-between h-auto"
                >
                  <div className="flex items-center gap-2 font-semibold">
                    <Terminal className="w-5 h-5" />
                    Conversion Logs {jobId && <span className="text-xs text-muted-foreground">({jobId})</span>}
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
                  id="convert-logs-body"
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
      </div>
    </div>
  );
}
