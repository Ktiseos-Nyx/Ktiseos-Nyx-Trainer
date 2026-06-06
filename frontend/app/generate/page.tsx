'use client';

/**
 * Generate page — our Next.js UI at route /generate (COMFY-3).
 *
 * NOTE: deliberately NOT /comfyui. server.js reverse-proxies /comfyui/* to the
 * ComfyUI service, so a page at /comfyui gets shadowed by the proxy and renders
 * literal ComfyUI (see BETA_PLANNING GEN-1). The GenerateUI this renders still
 * talks to ComfyUI through that /comfyui proxy.
 *
 * Connection states:
 *   Never connected → show DisconnectedState / ConnectingState cards as before.
 *   Once connected  → keep GenerateUI mounted permanently so prompts and
 *                     settings survive reconnects. Status changes fire toasts
 *                     instead of swapping the whole UI out.
 */

import { useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';
import { Loader2, RefreshCw, Wand2, AlertCircle, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ComfyConnectionStatus } from '@/components/comfy/ComfyConnectionStatus';
import { GenerateUI } from '@/components/comfy/GenerateUI';
import { useComfyConnection } from '@/lib/comfy';
import BorderGlow from '@/components/BorderGlow';
import { NoiseTexture } from '@/components/ui/noise-texture';

// ─── Pre-connect status cards ─────────────────────────────────────────────────

function DisconnectedState({
  onRetry,
  isError,
}: {
  onRetry: () => void;
  isError: boolean;
}) {
  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <div className="w-full max-w-md">
        <BorderGlow
          animated
          colors={['#a855f7', '#ec4899', '#38bdf8']}
          glowColor="280 70 70"
          glowIntensity={0.8}
          borderRadius={24}
          glowRadius={32}
          backgroundColor="hsl(var(--card))"
        >
          <div className="flex flex-col items-center gap-5 py-8 px-6 text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-muted/60">
              {isError ? (
                <AlertCircle className="h-8 w-8 text-red-400" />
              ) : (
                <Wand2 className="h-8 w-8 text-muted-foreground" />
              )}
            </div>

            <div className="space-y-1.5">
              <h2 className="text-lg font-semibold">
                {isError ? 'Connection error' : 'ComfyUI is not running'}
              </h2>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {isError
                  ? 'Could not reach ComfyUI. Check that it started correctly and try again.'
                  : 'Start ComfyUI on your machine (default port 8188) and this page will connect automatically.'}
              </p>
            </div>

            <div className="flex flex-col gap-2 w-full">
              <Button onClick={onRetry} className="w-full gap-2">
                <RefreshCw className="h-4 w-4" />
                Retry connection
              </Button>
              <Button variant="ghost" size="sm" asChild className="gap-1.5 text-muted-foreground">
                <a
                  href="https://github.com/comfyanonymous/ComfyUI#installing"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                  ComfyUI install guide
                </a>
              </Button>
            </div>

            <p className="text-xs text-muted-foreground/60">
              ComfyUI URL can be changed in{' '}
              <a href="/settings" className="underline underline-offset-2 hover:text-muted-foreground">
                Settings
              </a>
            </p>
          </div>
        </BorderGlow>
      </div>
    </div>
  );
}

function ConnectingState() {
  return (
    <div className="flex flex-1 items-center justify-center gap-3 text-muted-foreground">
      <Loader2 className="h-5 w-5 animate-spin" />
      <span className="text-sm">Connecting to ComfyUI…</span>
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function GeneratePage() {
  const {
    status, connect, queueRemaining,
    submitPrompt, interrupt, currentPromptId, currentNode, progress,
  } = useComfyConnection();

  // Once we've connected at least once, keep GenerateUI mounted forever.
  const [hasEverConnected, setHasEverConnected] = useState(false);
  useEffect(() => {
    if (status === 'connected') setHasEverConnected(true);
  }, [status]);

  // Toast on status transitions after first connect so the user knows
  // what's happening without losing their prompts and settings.
  const prevStatusRef = useRef<string | null>(null);
  useEffect(() => {
    const prev = prevStatusRef.current;
    prevStatusRef.current = status;

    // Skip the very first render (no previous state to compare).
    if (prev === null || prev === status) return;

    if (status === 'connected' && prev !== 'connecting') {
      toast.success('ComfyUI reconnected');
    } else if (status === 'disconnected' && prev === 'connected') {
      toast.warning('ComfyUI disconnected', {
        description: 'Your prompts and settings are preserved. Reconnecting…',
      });
    } else if (status === 'error' && prev === 'connected') {
      toast.error('ComfyUI connection lost', {
        description: 'Check that ComfyUI is still running.',
      });
    }
  }, [status]);

  return (
    <div className="relative flex h-[calc(100vh-4rem)] flex-col overflow-hidden">
      <NoiseTexture noiseOpacity={0.25} frequency={0.65} />
      {/* Page header */}
      <div className="flex items-center justify-between border-b border-border/40 px-6 py-3 shrink-0">
        <div className="flex items-center gap-3">
          <Wand2 className="h-5 w-5 text-muted-foreground" />
          <h1 className="text-base font-semibold">Generate</h1>
        </div>
        <ComfyConnectionStatus status={status} queueRemaining={queueRemaining} />
      </div>

      {/* GenerateUI stays mounted once we've connected so state is preserved.
          Before first connect, show the normal status cards. */}
      {hasEverConnected ? (
        <GenerateUI
          submitPrompt={submitPrompt}
          interrupt={interrupt}
          currentPromptId={currentPromptId}
          currentNode={currentNode}
          progress={progress}
          queueRemaining={queueRemaining}
          isConnected={status === 'connected'}
        />
      ) : (
        <>
          {status === 'connecting' && <ConnectingState />}
          {(status === 'disconnected' || status === 'error') && (
            <DisconnectedState onRetry={connect} isError={status === 'error'} />
          )}
        </>
      )}
    </div>
  );
}
