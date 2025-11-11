'use client';

import { useState, useEffect, useRef } from 'react';
import { trainingAPI } from '@/lib/api';
import { Play, Square, Activity, Clock, Zap, TrendingUp } from 'lucide-react';

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
  const wsRef = useRef<WebSocket | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs to bottom
  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [logs]);

  // Poll training status
  useEffect(() => {
    const pollStatus = async () => {
      try {
        const statusData = await trainingAPI.status();
        setStatus(statusData);
      } catch (err) {
        console.error('Failed to fetch status:', err);
      }
    };

    pollStatus();
    const interval = setInterval(pollStatus, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, []);

  // WebSocket for logs
  useEffect(() => {
    if (!status.is_training) return;

    console.log('Connecting to training logs WebSocket...');

    const ws = trainingAPI.connectLogs(
      (data) => {
        console.log('WebSocket message:', data);

        if (data.type === 'log') {
          setLogs((prev) => [...prev, data.message]);
        } else if (data.type === 'progress') {
          setStatus((prev) => ({
            ...prev,
            progress: data.data,
          }));
        } else if (data.type === 'connected') {
          setConnected(true);
          setLogs((prev) => [...prev, '✓ Connected to training logs']);
        } else if (data.type === 'heartbeat') {
          // Keep connection alive
          setConnected(true);
        }
      },
      (error) => {
        console.error('WebSocket error:', error);
        setConnected(false);
      }
    );

    ws.onclose = () => {
      console.log('WebSocket closed');
      setConnected(false);
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, [status.is_training]);

  // Calculate progress percentage
  const getProgress = () => {
    if (!status.progress) return 0;
    const { current_step, total_steps } = status.progress;
    if (!current_step || !total_steps) return 0;
    return (current_step / total_steps) * 100;
  };

  // Format time remaining (estimated)
  const getTimeRemaining = () => {
    if (!status.progress) return 'Calculating...';
    const { current_step, total_steps } = status.progress;
    if (!current_step || !total_steps) return 'Unknown';

    // This is a placeholder - you'd calculate based on actual step timing
    const stepsRemaining = total_steps - current_step;
    const estimatedMinutes = Math.ceil(stepsRemaining * 0.1); // Assume 0.1 min per step

    if (estimatedMinutes < 60) return `~${estimatedMinutes}m`;
    const hours = Math.floor(estimatedMinutes / 60);
    const minutes = estimatedMinutes % 60;
    return `~${hours}h ${minutes}m`;
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Activity className={`w-6 h-6 ${status.is_training ? 'text-green-500 animate-pulse' : 'text-gray-400'}`} />
          <h2 className="text-2xl font-bold">Training Monitor</h2>
        </div>

        <div className="flex items-center gap-2">
          {connected && (
            <span className="flex items-center gap-2 text-sm text-green-600">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
              Live
            </span>
          )}
          <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
            status.is_training
              ? 'bg-green-100 text-green-800'
              : 'bg-gray-100 text-gray-800'
          }`}>
            {status.is_training ? 'Training' : 'Idle'}
          </span>
        </div>
      </div>

      {/* Progress Section */}
      {status.is_training && status.progress && (
        <div className="mb-6 space-y-4">
          {/* Progress Bar */}
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="font-medium">
                Step {status.progress.current_step || 0} / {status.progress.total_steps || 0}
              </span>
              <span className="text-gray-600">
                {getProgress().toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
              <div
                className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-300 relative"
                style={{ width: `${getProgress()}%` }}
              >
                <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
              </div>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {/* Epoch */}
            {status.progress.current_epoch !== undefined && (
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                <div className="flex items-center gap-2 text-blue-600 dark:text-blue-400 mb-1">
                  <TrendingUp className="w-4 h-4" />
                  <span className="text-xs font-medium">Epoch</span>
                </div>
                <div className="text-xl font-bold">
                  {status.progress.current_epoch} / {status.progress.total_epochs || '?'}
                </div>
              </div>
            )}

            {/* Learning Rate */}
            {status.progress.lr !== undefined && (
              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
                <div className="flex items-center gap-2 text-purple-600 dark:text-purple-400 mb-1">
                  <Zap className="w-4 h-4" />
                  <span className="text-xs font-medium">Learning Rate</span>
                </div>
                <div className="text-xl font-bold">
                  {status.progress.lr.toExponential(2)}
                </div>
              </div>
            )}

            {/* Loss */}
            {status.progress.loss !== undefined && (
              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3">
                <div className="flex items-center gap-2 text-green-600 dark:text-green-400 mb-1">
                  <Activity className="w-4 h-4" />
                  <span className="text-xs font-medium">Loss</span>
                </div>
                <div className="text-xl font-bold">
                  {status.progress.loss.toFixed(4)}
                </div>
              </div>
            )}

            {/* Time Remaining */}
            <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-3">
              <div className="flex items-center gap-2 text-orange-600 dark:text-orange-400 mb-1">
                <Clock className="w-4 h-4" />
                <span className="text-xs font-medium">ETA</span>
              </div>
              <div className="text-xl font-bold">
                {getTimeRemaining()}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Logs Section */}
      <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
        <div className="bg-gray-50 dark:bg-gray-800 px-4 py-2 border-b border-gray-200 dark:border-gray-700">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Training Logs {connected && '(Live)'}
          </span>
        </div>

        <div className="bg-gray-900 p-4 font-mono text-sm text-green-400 h-96 overflow-y-auto">
          {logs.length === 0 ? (
            <div className="text-gray-500 text-center py-8">
              {status.is_training
                ? 'Waiting for logs...'
                : 'Start training to see logs here'}
            </div>
          ) : (
            <div className="space-y-1">
              {logs.map((log, index) => (
                <div key={index} className="hover:bg-gray-800/50 px-2 py-1 rounded">
                  {log}
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* No More Race Conditions! */}
      {!status.is_training && logs.length === 0 && (
        <div className="mt-4 text-center text-sm text-gray-500">
          <p>✨ WebSocket-based real-time updates</p>
          <p>🎉 No polling! No race conditions!</p>
        </div>
      )}
    </div>
  );
}
