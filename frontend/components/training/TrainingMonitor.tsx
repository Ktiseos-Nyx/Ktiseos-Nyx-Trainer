'use client';

import { useState, useEffect, useRef } from 'react';
import { trainingAPI } from '@/lib/api';
import { Activity, Save, Clock, Zap, TrendingUp } from 'lucide-react';

interface TrainingStatus {
  is_training: boolean;
  progress?: {
    current_step?: number;
    total_steps?: number;
    current_epoch?: number;
    total_epochs?: number;
    loss?: number;
    lr?: number;
  };
}

export default function TrainingMonitor() {
  const [status, setStatus] = useState<TrainingStatus>({ is_training: false });
  const [logs, setLogs] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);

  // ✅ FIX 1: Initialize as null (safe for server)
  const [jobId, setJobId] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // ✅ FIX 2: Load from URL query parameter, then localStorage after component mounts (browser only)
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const urlParams = new URLSearchParams(window.location.search);
      const urlJobId = urlParams.get('job');

      if (urlJobId) {
        setJobId(urlJobId);
        localStorage.setItem('current_training_job_id', urlJobId);
      } else {
        const savedJobId = localStorage.getItem('current_training_job_id');
        if (savedJobId) {
          setJobId(savedJobId);
        }
      }
    }
  }, []);

  // Auto-scroll logs to bottom
  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [logs]);

  // Initial check on mount (for page refreshes during active training)
  useEffect(() => {
    // Wait for jobId to be loaded
    if (!jobId) return;

    const checkInitialStatus = async (currentJobId: string) => {
      try {
        const statusData = await trainingAPI.status(currentJobId);
        setStatus(statusData);
      } catch (err) {
        // No training running - stay idle
        setStatus({ is_training: false });
      }
    };

    checkInitialStatus(jobId);

    // Listen for training start event from TrainingConfig
    const handleTrainingStart = (event: any) => {
      // Use event detail, fallback to localStorage if window exists
      const newJobId = event.detail?.jobId || (typeof window !== 'undefined' ? localStorage.getItem('current_training_job_id') : null);
      if (newJobId) {
        setJobId(newJobId);
        checkInitialStatus(newJobId);
      }
    };

    if (typeof window !== 'undefined') {
      window.addEventListener('training-started', handleTrainingStart);
    }

    return () => {
      if (typeof window !== 'undefined') {
        window.removeEventListener('training-started', handleTrainingStart);
      }
    };
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

  // WebSocket for logs
  useEffect(() => {
    if (!status.is_training || !jobId) return;

    console.log(`Connecting to training logs WebSocket for job ${jobId}...`);

    const ws = trainingAPI.connectLogs(
      jobId,
      (data) => {
        if (data.type === 'log') {
          setLogs((prev) => [...prev, data.message as string]);
        } else if (data.type === 'progress') {
          setStatus((prev) => ({
            ...prev,
            progress: data.data as any,
          }));
        } else if (data.type === 'connected') {
          setConnected(true);
          setLogs((prev) => [...prev, '✓ Connected to training logs']);
        } else if (data.type === 'heartbeat') {
          setConnected(true);
        }
      },
      (error) => {
        console.error('WebSocket error:', error);
        setConnected(false);
      }
    );

    ws.onclose = () => {
      setConnected(false);
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, [status.is_training, jobId]);

  const getProgress = () => {
    if (!status.progress) return 0;
    const { current_step, total_steps } = status.progress;
    if (!current_step || !total_steps) return 0;
    return (current_step / total_steps) * 100;
  };

  const getTimeRemaining = () => {
    if (!status.progress) return 'Calculating...';
    const { current_step, total_steps } = status.progress;
    if (!current_step || !total_steps) return 'Unknown';

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
            {status.progress.current_epoch !== undefined && (
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
            {status.progress.lr !== undefined && (
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
            {status.progress.loss !== undefined && (
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

        <div className="bg-background  p-4 font-mono text-sm text-green-400 h-96 overflow-y-auto">
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
          <p>WebSocket-based real-time updates</p>
        </div>
      )}
    </div>
  );
}
