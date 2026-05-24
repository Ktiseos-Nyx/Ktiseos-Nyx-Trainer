'use client';

/**
 * useComfyConnection — React hook for ComfyUI WebSocket state.
 *
 * Manages a single WebSocket connection per component tree.
 * Tracks connection status, queue depth, and per-step progress so
 * the UI can reflect what ComfyUI is doing in real time.
 *
 * Usage:
 *   const { status, progress, queueRemaining, submit, interrupt } = useComfyConnection();
 *
 * The hook reconnects automatically on disconnect (3 s delay).
 * Call `disconnect()` to stop reconnecting (e.g. on settings page when the
 * user clears the ComfyUI URL).
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { comfyClient, type ComfyWsHandle } from './client';
import type {
  ComfyConnectionState,
  ComfyConnectionStatus,
  ComfyPromptRequest,
  ComfyWsExecuted,
  ComfyWsError,
  ComfyWsExecuting,
  ComfyWsExecutionStart,
  ComfyWsMessage,
  ComfyWsProgress,
  ComfyWsStatus,
} from './types';
import { buildTxt2ImgWorkflow, type Txt2ImgParams } from './workflows';

// ─── Client ID ────────────────────────────────────────────────────────────────

/**
 * Generate or retrieve a stable client ID for this browser session.
 * Stored in sessionStorage so it's stable across re-renders but refreshes
 * on a new tab/window (avoids stale subscriptions).
 */
function getClientId(): string {
  const key = 'knx-comfy-client-id';
  if (typeof sessionStorage === 'undefined') return crypto.randomUUID();
  let id = sessionStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem(key, id);
  }
  return id;
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

const INITIAL_STATE: ComfyConnectionState = {
  status: 'disconnected',
  queueRemaining: 0,
  currentPromptId: null,
  currentNode: null,
  progress: null,
};

export interface UseComfyConnectionReturn extends ComfyConnectionState {
  /** Connect (or reconnect) to ComfyUI. Called automatically on mount. */
  connect: () => void;
  /** Deliberately disconnect and stop auto-reconnect. */
  disconnect: () => void;
  /**
   * Submit a txt2img job. Injects the stable clientId automatically.
   * Returns the prompt_id assigned by ComfyUI.
   */
  submitTxt2Img: (params: Txt2ImgParams) => Promise<string>;
  /**
   * Submit an arbitrary pre-built workflow prompt.
   * Returns the prompt_id assigned by ComfyUI.
   */
  submitPrompt: (req: Omit<ComfyPromptRequest, 'client_id'>) => Promise<string>;
  /** Cancel the currently running prompt. */
  interrupt: () => Promise<void>;
  /** The stable client ID used for this session's WebSocket. */
  clientId: string;
}

export function useComfyConnection(): UseComfyConnectionReturn {
  const [state, setState] = useState<ComfyConnectionState>(INITIAL_STATE);
  const wsHandleRef = useRef<ComfyWsHandle | null>(null);
  const clientId = useRef(getClientId()).current;

  const handleMessage = useCallback((msg: ComfyWsMessage) => {
    switch (msg.type) {
      case 'status': {
        const { data } = msg as ComfyWsStatus;
        setState(s => ({ ...s, queueRemaining: data.status.exec_info.queue_remaining }));
        break;
      }

      case 'execution_start': {
        const { data } = msg as ComfyWsExecutionStart;
        setState(s => ({
          ...s,
          currentPromptId: data.prompt_id,
          currentNode: null,
          progress: null,
        }));
        break;
      }

      case 'executing': {
        const { data } = msg as ComfyWsExecuting;
        // node: null signals the prompt is fully done
        if (data.node === null) {
          setState(s => ({
            ...s,
            currentPromptId: null,
            currentNode: null,
            progress: null,
          }));
        } else {
          setState(s => ({ ...s, currentNode: data.node }));
        }
        break;
      }

      case 'progress': {
        const { data } = msg as ComfyWsProgress;
        setState(s => ({
          ...s,
          progress: { value: data.value, max: data.max },
        }));
        break;
      }

      case 'executed':
        // Individual node output — UI can handle via onMessage callback if needed
        void (msg as ComfyWsExecuted);
        break;

      case 'execution_error': {
        void (msg as ComfyWsError);
        setState(s => ({
          ...s,
          currentPromptId: null,
          currentNode: null,
          progress: null,
        }));
        break;
      }

      default:
        break;
    }
  }, []);

  const connect = useCallback(() => {
    // Close any existing socket before reconnecting
    wsHandleRef.current?.close();

    setState(s => ({ ...s, status: 'connecting' as ComfyConnectionStatus }));

    wsHandleRef.current = comfyClient.connectWebSocket({
      clientId,
      onMessage: handleMessage,
      onOpen: () => setState(s => ({ ...s, status: 'connected' })),
      onClose: () => setState(s => ({ ...s, status: 'disconnected' })),
      onError: () => setState(s => ({ ...s, status: 'error' })),
    });
  }, [clientId, handleMessage]);

  const disconnect = useCallback(() => {
    wsHandleRef.current?.close();
    wsHandleRef.current = null;
    setState(INITIAL_STATE);
  }, []);

  const submitPrompt = useCallback(
    async (req: Omit<ComfyPromptRequest, 'client_id'>): Promise<string> => {
      const result = await comfyClient.submitPrompt({ ...req, client_id: clientId });
      return result.prompt_id;
    },
    [clientId],
  );

  const submitTxt2Img = useCallback(
    async (params: Txt2ImgParams): Promise<string> => {
      const workflow = buildTxt2ImgWorkflow(params);
      return submitPrompt({ prompt: workflow });
    },
    [submitPrompt],
  );

  const interrupt = useCallback(async () => {
    await comfyClient.interrupt();
  }, []);

  // Auto-connect on mount; disconnect on unmount
  useEffect(() => {
    connect();
    return () => {
      wsHandleRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    ...state,
    connect,
    disconnect,
    submitPrompt,
    submitTxt2Img,
    interrupt,
    clientId,
  };
}
