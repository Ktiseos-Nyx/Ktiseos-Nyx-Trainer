'use client';

/**
 * ComfyUI generate page (COMFY-3).
 *
 * Renders connection-aware states:
 *   disconnected / error → BorderGlow card with setup instructions + retry
 *   connecting           → spinner while WebSocket handshakes
 *   connected            → generate UI (full UI in subsequent tickets)
 */

import { Loader2, RefreshCw, Wand2, AlertCircle, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ComfyConnectionStatus } from '@/components/comfy/ComfyConnectionStatus';
import { GenerateUI } from '@/components/comfy/GenerateUI';
import { useComfyConnection } from '@/lib/comfy';
import BorderGlow from '@/components/BorderGlow';

// ─── Disconnected state ───────────────────────────────────────────────────────

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

// ─── Connecting state ─────────────────────────────────────────────────────────

function ConnectingState() {
  return (
    <div className="flex flex-1 items-center justify-center gap-3 text-muted-foreground">
      <Loader2 className="h-5 w-5 animate-spin" />
      <span className="text-sm">Connecting to ComfyUI…</span>
    </div>
  );
}


// ─── Page ─────────────────────────────────────────────────────────────────────

export default function ComfyUIPage() {
  const {
    status, connect, queueRemaining,
    submitPrompt, interrupt, currentPromptId, currentNode, progress,
  } = useComfyConnection();

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col">
      {/* Page header */}
      <div className="flex items-center justify-between border-b border-border/40 px-6 py-3 shrink-0">
        <div className="flex items-center gap-3">
          <Wand2 className="h-5 w-5 text-muted-foreground" />
          <h1 className="text-base font-semibold">Generate</h1>
        </div>
        <ComfyConnectionStatus status={status} queueRemaining={queueRemaining} />
      </div>

      {/* Body */}
      {status === 'connecting' && <ConnectingState />}
      {(status === 'disconnected' || status === 'error') && (
        <DisconnectedState onRetry={connect} isError={status === 'error'} />
      )}
      {status === 'connected' && (
        <GenerateUI
          submitPrompt={submitPrompt}
          interrupt={interrupt}
          currentPromptId={currentPromptId}
          currentNode={currentNode}
          progress={progress}
          queueRemaining={queueRemaining}
        />
      )}
    </div>
  );
}
