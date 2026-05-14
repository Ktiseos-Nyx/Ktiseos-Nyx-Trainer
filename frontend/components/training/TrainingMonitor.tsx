'use client';

import { useState, useEffect, useRef } from 'react';
import { trainingAPI, LogPoller } from '@/lib/api';
import { Activity, Clock, Zap, TrendingUp } from 'lucide-react';

interface TrainingStatus {
  is_training: boolean;
  error?: string;
  progress?: {
    current_step?: number;
    total_steps?: number;
    current_epoch?: number;
    total_epochs?: number;
    loss?: number;
    lr?: number;
    eta_seconds?: number;
    progress_percent?: number;
  };
}

/**
 * Real-time training monitor that polls job status and streams logs.
 * Uses HTTP polling instead of WebSockets to survive Caddy proxy disconnects.
 * Job ID is sourced from the `?job=` query param or localStorage fallback.
 */
export default function TrainingMonitor() {
  const [status, setStatus] = useState<TrainingStatus>({ is_training: false });
  const [logs, setLogs] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);

  // Initialize jobId from URL query param or localStorage (browser only)
  const [jobId, setJobId] = useState<string | null>(() => {
    if (typeof window === 'undefined') return null;
    const urlParams = new URLSearchParams(window.location.search);
    const urlJobId = urlParams.get('job');
    if (urlJobId) {
      localStorage.setItem('current_training_job_id', urlJobId);
      return urlJobId;
    }
    return localStorage.getItem('current_training_job_id');
  });

  const logPollerRef = useRef<LogPoller | null>(null);
  const drainTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const MAX_LOGS = 1000;

  // Updated synchronously during render so effect cleanups always see the
  // latest value — useEffect deps are stale closures, refs are not.
  const isTrainingRef = useRef(status.is_training);
  isTrainingRef.current = status.is_training;

  // Auto-scroll logs to bottom
  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [logs]);

  // Listen for training start event from TrainingConfig (ALWAYS active)
  // This must be a separate effect with no jobId dependency so the listener
  // stays registered even when jobId is null (e.g. after clearing a stale ID).
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const handleTrainingStart = (event: CustomEvent<{ jobId?: string }>) => {
      const newJobId = event.detail?.jobId || localStorage.getItem('current_training_job_id');
      if (newJobId) {
        console.log(`TrainingMonitor: received training-started event for ${newJobId}`);
        setLogs([]); // Clear previous logs
        setJobId(newJobId);
      }
    };

    window.addEventListener('training-started', handleTrainingStart);
    return () => {
      window.removeEventListener('training-started', handleTrainingStart);
    };
  }, []);

  // Check status when jobId changes (on mount with stored ID, or after training-started event)
  useEffect(() => {
    if (!jobId) return;

    const checkInitialStatus = async () => {
      try {
        const statusData = await trainingAPI.status(jobId);
        setStatus(statusData);

        // If job is done (completed/failed/cancelled), clear stale ID
        if (!statusData.is_training) {
          localStorage.removeItem('current_training_job_id');
        }
      } catch (err) {
        // Job not found (e.g. backend restarted) - clear stale job ID
        setStatus({ is_training: false });
        localStorage.removeItem('current_training_job_id');
        setJobId(null);
      }
    };

    checkInitialStatus();
  }, [jobId]);

  // Poll training status (only when training is active)
  useEffect(() => {
    // Don't poll if not training or no job_id
    if (!status.is_training || !jobId) return;

    const pollStatus = async () => {
      if (typeof document !== 'undefined' && document.hidden) return;

      try {
        const statusData = await trainingAPI.status(jobId);
        setStatus(statusData);
      } catch (err: any) {
        setStatus({ is_training: false });
        if (typeof window !== 'undefined') {
          localStorage.removeItem('current_training_job_id');
        }
      }
    };

    const interval = setInterval(pollStatus, 2000);

    const handleVisibilityChange = () => {
      if (typeof document !== 'undefined' && !document.hidden) {
        pollStatus();
      }
    };

    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', handleVisibilityChange);
    }

    return () => {
      clearInterval(interval);
      if (typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', handleVisibilityChange);
      }
    };
  }, [status.is_training, jobId]);

  // HTTP polling for logs (replaces WebSocket which breaks through Caddy proxy)
  useEffect(() => {
    if (!status.is_training || !jobId) return;

    console.log(`Starting log polling for job ${jobId}...`);
    setConnected(true);

    const poller = trainingAPI.pollLogs(
      jobId,
      (data) => {
        if (data.type === 'log' && data.log) {
          const msg = data.log;
          setLogs((prev) => {
            const next = [...prev, msg];
            return next.length > MAX_LOGS ? next.slice(-MAX_LOGS) : next;
          });
        } else if (data.type === 'progress' && data.progress != null) {
          setStatus((prev) => ({
            ...prev,
            progress: {
              ...prev.progress,
              progress_percent: data.progress as number,
            },
          }));
        } else if (data.type === 'step_progress') {
          setStatus((prev) => ({
            ...prev,
            progress: {
              ...prev.progress,
              ...(data.step_num != null && { current_step: data.step_num }),
              ...(data.total_steps != null && { total_steps: data.total_steps }),
              ...(data.current_epoch != null && { current_epoch: data.current_epoch }),
              ...(data.total_epochs != null && { total_epochs: data.total_epochs }),
              ...(data.loss != null && { loss: data.loss }),
              ...(data.lr != null && { lr: data.lr }),
              ...(data.eta_seconds != null && { eta_seconds: data.eta_seconds }),
            },
          }));
        } else if (data.type === 'status') {
          if (data.status === 'completed') {
            setStatus({ is_training: false });
            setLogs((prev) => [...prev, '--- Training completed! ---']);
            setConnected(false);
            localStorage.removeItem('current_training_job_id');
          } else if (data.status === 'failed') {
            const errorMsg = typeof data['error'] === 'string' ? data['error'] : 'Unknown error';
            setStatus({ is_training: false, error: errorMsg });
            setLogs((prev) => [...prev, `--- Training FAILED: ${errorMsg} ---`]);
            setConnected(false);
            localStorage.removeItem('current_training_job_id');
          }
        }
      },
      (error) => {
        console.error('Log polling error:', error);
        setConnected(false);
      }
    );

    logPollerRef.current = poller;

    return () => {
      if (drainTimeoutRef.current) {
        clearTimeout(drainTimeoutRef.current);
        drainTimeoutRef.current = null;
      }

      if (!isTrainingRef.current) {
        // The status poller declared training done before the log poller could
        // drain the final epochs. Let the log poller run — it self-terminates
        // when the log endpoint returns 'completed' or 'failed'.
        // Safety net: force-stop after 15s if the completion signal never arrives.
        drainTimeoutRef.current = setTimeout(() => {
          poller.stop();
          if (logPollerRef.current === poller) setConnected(false);
        }, 15000);
      } else {
        // Job changed or component unmounted — stop immediately.
        poller.stop();
        setConnected(false);
      }
    };
  }, [status.is_training, jobId]);

  /** Returns 0–100 progress percentage, preferring the backend-parsed value over the step-based estimate. */
  const getProgress = () => {
    if (!status.progress) return 0;
    if (status.progress.progress_percent != null) return status.progress.progress_percent;
    const { current_step, total_steps } = status.progress;
    if (current_step == null || total_steps == null || total_steps === 0) return 0;
    return (current_step / total_steps) * 100;
  };

  /** Returns a human-readable ETA string, preferring tqdm's parsed eta_seconds over a rough step-based estimate. */
  const getTimeRemaining = () => {
    if (!status.progress) return 'Calculating...';
    const { eta_seconds, current_step, total_steps } = status.progress;

    // Use tqdm's parsed ETA if available — it's far more accurate than our estimate
    if (eta_seconds != null) {
      if (eta_seconds < 60) return `~${eta_seconds}s`;
      const h = Math.floor(eta_seconds / 3600);
      const m = Math.floor((eta_seconds % 3600) / 60);
      return h > 0 ? `~${h}h ${m}m` : `~${m}m`;
    }

    if (current_step == null || total_steps == null || total_steps === 0) return 'Unknown';
    const stepsRemaining = total_steps - current_step;
    const estimatedMinutes = Math.ceil(stepsRemaining * 0.1);
    if (estimatedMinutes < 60) return `~${estimatedMinutes}m`;
    const hours = Math.floor(estimatedMinutes / 60);
    const minutes = estimatedMinutes % 60;
    return `~${hours}h ${minutes}m`;
  };

  return (
    <div className="rounded-xl border bg-card text-card-foreground shadow w-full">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Activity className={`w-6 h-6 ${status.is_training ? 'text-green-400 animate-pulse' : 'text-muted-foreground'}`} />
          <h2 className="text-2xl font-bold text-foreground">Training Monitor</h2>
        </div>
        <div className="flex items-center gap-2">
          {connected && (
            <span className="flex items-center gap-2 text-sm text-green-400">
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
              Live
            </span>
          )}
          <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
            status.is_training
              ? 'bg-green-300/20 text-green-400 border border-green-500/30'
              : 'bg-muted text-muted-foreground border border-border'
          }`}>
            {status.is_training ? 'Training' : 'Idle'}
          </span>
        </div>
      </div>

      {status.is_training && status.progress && (
        <div className="mb-6 space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="font-medium text-foreground">
                Step {status.progress.current_step || 0} / {status.progress.total_steps || 0}
              </span>
              <span className="text-muted-foreground">
                {getProgress().toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-muted rounded-full h-3 overflow-hidden">
              <div
                className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-300 relative"
                style={{ width: `${getProgress()}%` }}
              >
                <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {status.progress.current_epoch != null && (
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
                <div className="flex items-center gap-2 text-blue-400 mb-1">
                  <TrendingUp className="w-4 h-4" />
                  <span className="text-xs font-medium">Epoch</span>
                </div>
                <div className="text-xl font-bold text-white">
                  {status.progress.current_epoch} / {status.progress.total_epochs || '?'}
                </div>
              </div>
            )}
            {status.progress.lr != null && (
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
                <div className="flex items-center gap-2 text-purple-400 mb-1">
                  <Zap className="w-4 h-4" />
                  <span className="text-xs font-medium">Learning Rate</span>
                </div>
                <div className="text-xl font-bold text-white">
                  {status.progress.lr.toExponential(2)}
                </div>
              </div>
            )}
            {status.progress.loss != null && (
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
                <div className="flex items-center gap-2 text-green-400 mb-1">
                  <Activity className="w-4 h-4" />
                  <span className="text-xs font-medium">Loss</span>
                </div>
                <div className="text-xl font-bold text-white">
                  {status.progress.loss.toFixed(4)}
                </div>
              </div>
            )}
            <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-3">
              <div className="flex items-center gap-2 text-orange-400 mb-1">
                <Clock className="w-4 h-4" />
                <span className="text-xs font-medium">ETA</span>
              </div>
              <div className="text-xl font-bold text-white">
                {getTimeRemaining()}
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="border border-border rounded-lg overflow-hidden">
        <div className="bg-muted px-4 py-2 border-b border-border">
          <span className="text-sm font-medium text-foreground">
            Training Logs {connected && <span className="text-green-400">(Live)</span>}
          </span>
        </div>

        <div className="bg-background p-4 font-mono text-sm text-green-400 h-96 overflow-y-auto">
          {logs.length === 0 ? (
            <div className="text-muted-foreground text-center py-8">
              {status.is_training
                ? 'Waiting for logs...'
                : 'Start training to see logs here'}
            </div>
          ) : (
            <div className="space-y-1">
              {logs.map((log, index) => (
                <div key={index} className="hover:bg-accent px-2 py-1 rounded">
                  {log}
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          )}
        </div>
      </div>

      {!status.is_training && logs.length === 0 && (
        <div className="mt-4 text-center text-sm text-muted-foreground">
          <p>Real-time log updates via polling</p>
        </div>
      )}
    </div>
  );
}
