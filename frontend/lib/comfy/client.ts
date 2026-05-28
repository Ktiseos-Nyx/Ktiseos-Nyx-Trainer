'use client';

/**
 * ComfyUI API client.
 *
 * All requests go through the `/comfyui` reverse proxy in server.js, which
 * strips the prefix and forwards to ComfyUI on localhost:8188 (or COMFYUI_PORT).
 *
 * WebSocket connects to `/comfyui/ws?clientId=<uuid>` — the proxy rewrites
 * this to `ws://127.0.0.1:8188/ws?clientId=<uuid>`.
 *
 * Usage:
 *   import { comfyClient } from '@/lib/comfy/client';
 *   const stats = await comfyClient.getSystemStats();
 */

import type {
  ComfyHistory,
  ComfyHistoryEntry,
  ComfyObjectInfo,
  ComfyOutputFile,
  ComfyPromptRequest,
  ComfyPromptResponse,
  ComfyQueueStatus,
  ComfySystemStats,
  ComfyWsMessage,
} from './types';

// ─── Config ───────────────────────────────────────────────────────────────────

/** Base URL for all ComfyUI HTTP requests. Proxied through server.js. */
const BASE_URL = '/comfyui';

/** WebSocket base URL — same proxy path, protocol flipped. */
const WS_BASE_URL =
  typeof window !== 'undefined'
    ? `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/comfyui`
    : '';

// ─── Helpers ─────────────────────────────────────────────────────────────────

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`ComfyUI ${path} → ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ─── REST methods ─────────────────────────────────────────────────────────────

/** Submit a workflow prompt. Returns the assigned prompt_id. */
async function submitPrompt(body: ComfyPromptRequest): Promise<ComfyPromptResponse> {
  return request<ComfyPromptResponse>('/prompt', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

/** Interrupt the currently running prompt. */
async function interrupt(): Promise<void> {
  await request<unknown>('/interrupt', { method: 'POST' });
}

/** Fetch current queue state. */
async function getQueue(): Promise<ComfyQueueStatus> {
  return request<ComfyQueueStatus>('/queue');
}

/** Fetch full history, or a single entry by promptId. */
async function getHistory(): Promise<ComfyHistory>;
async function getHistory(promptId: string): Promise<ComfyHistoryEntry | null>;
async function getHistory(promptId?: string): Promise<ComfyHistory | ComfyHistoryEntry | null> {
  if (promptId) {
    const data = await request<Record<string, ComfyHistoryEntry>>(`/history/${promptId}`);
    return data[promptId] ?? null;
  }
  return request<ComfyHistory>('/history');
}

/** Clear history entries. Pass promptIds to remove specific entries, or omit to clear all. */
async function deleteHistory(promptIds?: string[]): Promise<void> {
  const body = promptIds ? { delete: promptIds } : { clear: true };
  await request<unknown>('/history', { method: 'DELETE', body: JSON.stringify(body) });
}

/** Fetch available node types and their input specs. */
async function getObjectInfo(): Promise<ComfyObjectInfo> {
  return request<ComfyObjectInfo>('/object_info');
}

/**
 * Fetch the list of model filenames in a ComfyUI model folder.
 *
 * Common folder names: `diffusion_models`, `checkpoints`, `text_encoders`,
 * `clip`, `vae`, `loras`, `upscale_models`, `unet`.
 *
 * Returns an empty array if the folder doesn't exist in this ComfyUI install.
 */
async function getModelFiles(folder: string): Promise<string[]> {
  return request<string[]>(`/models/${encodeURIComponent(folder)}`);
}

/**
 * Fetch all LoRA filenames from LoRA Manager's cache.
 *
 * Unlike ComfyUI's /models/loras, the LM cache includes every root in
 * extra_model_paths.yaml — including the training output/ directory.
 * Returns filenames (basename with extension) sorted alphabetically.
 *
 * Returns an empty array if LoRA Manager is not installed.
 */
async function getLmLoras(): Promise<string[]> {
  try {
    const data = await request<{ items: Array<{ file_path: string; file_name: string }> }>(
      '/api/lm/loras/list?page_size=9999'
    );
    return data.items
      .map(item => {
        const parts = item.file_path.replace(/\\/g, '/').split('/');
        return parts[parts.length - 1] ?? item.file_name;
      })
      .sort();
  } catch {
    return [];
  }
}

/** Fetch system stats (GPU VRAM, OS, Python version). */
async function getSystemStats(): Promise<ComfySystemStats> {
  return request<ComfySystemStats>('/system_stats');
}

/**
 * Build a URL to view or download a generated image.
 *
 * @param file  Output file descriptor from history/ws events.
 * @param preview  If true, return a lower-res preview (uses ComfyUI's view endpoint).
 */
function getImageUrl(file: ComfyOutputFile, preview = false): string {
  const params = new URLSearchParams({
    filename: file.filename,
    type: file.type,
    subfolder: file.subfolder,
  });
  if (preview) params.set('preview', '1');
  return `${BASE_URL}/view?${params.toString()}`;
}

/**
 * Ping ComfyUI to check if it's reachable.
 * Returns true only if the server responds with an OK status; false if it's
 * down, the proxy errors, or the endpoint returns a non-2xx status.
 */
async function ping(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/system_stats`, { method: 'HEAD' });
    return res.ok;
  } catch {
    return false;
  }
}

// ─── WebSocket ────────────────────────────────────────────────────────────────

export interface ComfyWsOptions {
  clientId: string;
  onMessage: (msg: ComfyWsMessage) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (err: Event) => void;
  /** Auto-reconnect delay in ms. Set to 0 to disable. Default: 3000. */
  reconnectDelay?: number;
}

export interface ComfyWsHandle {
  /** Close the WebSocket and stop any pending reconnect. */
  close: () => void;
  /** True if the socket is currently open. */
  readonly isOpen: boolean;
}

/**
 * Open a WebSocket connection to ComfyUI and return a handle to close it.
 *
 * Auto-reconnects after `reconnectDelay` ms unless closed deliberately.
 */
function connectWebSocket(opts: ComfyWsOptions): ComfyWsHandle {
  const { clientId, onMessage, onOpen, onClose, onError, reconnectDelay = 3000 } = opts;

  let ws: WebSocket | null = null;
  let deliberatelyClosed = false;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let open = false;

  function connect() {
    if (deliberatelyClosed) return;

    const url = `${WS_BASE_URL}/ws?clientId=${encodeURIComponent(clientId)}`;
    ws = new WebSocket(url);

    ws.onopen = () => {
      open = true;
      onOpen?.();
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data as string) as ComfyWsMessage;
        onMessage(msg);
      } catch {
        // Binary frames (image previews) — ignore
      }
    };

    ws.onclose = () => {
      open = false;
      onClose?.();
      if (!deliberatelyClosed && reconnectDelay > 0) {
        reconnectTimer = setTimeout(connect, reconnectDelay);
      }
    };

    ws.onerror = (err) => {
      onError?.(err);
    };
  }

  connect();

  return {
    close() {
      deliberatelyClosed = true;
      if (reconnectTimer != null) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      ws?.close();
      open = false;
    },
    get isOpen() {
      return open;
    },
  };
}

// ─── Export ───────────────────────────────────────────────────────────────────

export const comfyClient = {
  submitPrompt,
  interrupt,
  getQueue,
  getHistory,
  deleteHistory,
  getObjectInfo,
  getModelFiles,
  getLmLoras,
  getSystemStats,
  getImageUrl,
  ping,
  connectWebSocket,
} as const;
