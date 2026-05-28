'use client';

/**
 * ComfyConnectionStatus — small status pill for the ComfyUI page header.
 *
 * Shows a pulsing/static coloured dot with a text label reflecting the
 * current WebSocket connection state.
 */

import { cn } from '@/lib/utils';
import type { ComfyConnectionStatus as ConnectionStatus } from '@/lib/comfy';

interface Props {
  status: ConnectionStatus;
  queueRemaining?: number;
  className?: string;
}

const CONFIG: Record<
  ConnectionStatus,
  { dot: string; pulse: boolean; label: string }
> = {
  connected: {
    dot: 'bg-emerald-500',
    pulse: false,
    label: 'Connected',
  },
  connecting: {
    dot: 'bg-yellow-400',
    pulse: true,
    label: 'Connecting…',
  },
  disconnected: {
    dot: 'bg-zinc-500',
    pulse: false,
    label: 'Not running',
  },
  error: {
    dot: 'bg-red-500',
    pulse: false,
    label: 'Error',
  },
};

export function ComfyConnectionStatus({ status, queueRemaining, className }: Props) {
  const { dot, pulse, label } = CONFIG[status];

  const queueLabel =
    status === 'connected' && queueRemaining != null && queueRemaining > 0
      ? ` · ${queueRemaining} queued`
      : '';

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium',
        'border-border/50 bg-background/60 text-muted-foreground backdrop-blur',
        className,
      )}
    >
      <span className="relative flex h-2 w-2">
        {pulse && (
          <span
            className={cn(
              'absolute inline-flex h-full w-full animate-ping rounded-full opacity-75',
              dot,
            )}
          />
        )}
        <span className={cn('relative inline-flex h-2 w-2 rounded-full', dot)} />
      </span>
      {label}{queueLabel}
    </span>
  );
}
