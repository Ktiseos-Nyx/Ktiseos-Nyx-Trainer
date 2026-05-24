/**
 * ComfyUI API type definitions.
 *
 * Covers the full ComfyUI REST + WebSocket surface used by this client.
 * All types are derived from ComfyUI's documented API (comfyanonymous/ComfyUI).
 */

// ─── Workflow / Prompt ────────────────────────────────────────────────────────

/** A single node's input values inside a workflow prompt. */
export type ComfyNodeInputs = Record<string, unknown>;

/** A single node in a ComfyUI workflow prompt. */
export interface ComfyNode {
  class_type: string;
  inputs: ComfyNodeInputs;
  /** Optional display metadata (title, color). Not used by the backend. */
  _meta?: { title?: string };
}

/**
 * A ComfyUI workflow prompt — a map of node IDs to node definitions.
 * Node IDs are arbitrary strings (ComfyUI uses numeric strings: "1", "2", …).
 */
export type ComfyWorkflow = Record<string, ComfyNode>;

/** Body sent to POST /prompt. */
export interface ComfyPromptRequest {
  prompt: ComfyWorkflow;
  /** Client ID registered on the WebSocket connection. */
  client_id: string;
  /** Optional extra data passed through to the history entry. */
  extra_data?: Record<string, unknown>;
}

/** Response from POST /prompt. */
export interface ComfyPromptResponse {
  prompt_id: string;
  number: number;
  node_errors: Record<string, unknown>;
}

// ─── Queue ────────────────────────────────────────────────────────────────────

/** A single item in the queue — [number, promptId, prompt, extraData, outputs?] */
export type ComfyQueueItem = [number, string, ComfyWorkflow, Record<string, unknown>];

/** Response from GET /queue. */
export interface ComfyQueueStatus {
  queue_running: ComfyQueueItem[];
  queue_pending: ComfyQueueItem[];
}

// ─── History ──────────────────────────────────────────────────────────────────

/** A single output image/video from a history entry. */
export interface ComfyOutputFile {
  filename: string;
  /** Subfolder inside the ComfyUI output directory. */
  subfolder: string;
  /** "output" | "temp" | "input" */
  type: string;
}

/** Outputs keyed by node ID → output name → files. */
export type ComfyOutputs = Record<string, Record<string, ComfyOutputFile[]>>;

/** Status block inside a history entry. */
export interface ComfyExecutionStatus {
  status_str: 'success' | 'error';
  completed: boolean;
  messages: Array<[string, Record<string, unknown>]>;
}

/** A single history entry returned by GET /history/{promptId}. */
export interface ComfyHistoryEntry {
  prompt: [number, string, ComfyWorkflow, Record<string, unknown>];
  outputs: ComfyOutputs;
  status: ComfyExecutionStatus;
}

/** Response from GET /history — map of promptId → entry. */
export type ComfyHistory = Record<string, ComfyHistoryEntry>;

// ─── System Stats ─────────────────────────────────────────────────────────────

export interface ComfySystemStats {
  system: {
    os: string;
    python_version: string;
    embedded_python: boolean;
  };
  devices: Array<{
    name: string;
    type: 'cuda' | 'mps' | 'cpu';
    index: number;
    vram_total: number;
    vram_free: number;
    torch_vram_total: number;
    torch_vram_free: number;
  }>;
}

// ─── Object Info (node registry) ──────────────────────────────────────────────

export interface ComfyInputSpec {
  [inputName: string]: unknown;
}

export interface ComfyNodeInfo {
  input: { required?: ComfyInputSpec; optional?: ComfyInputSpec };
  output: string[];
  output_is_list: boolean[];
  output_name: string[];
  name: string;
  display_name: string;
  description: string;
  category: string;
  output_node: boolean;
}

/** Response from GET /object_info — map of class_type → node definition. */
export type ComfyObjectInfo = Record<string, ComfyNodeInfo>;

// ─── WebSocket messages ───────────────────────────────────────────────────────

export type ComfyWsMessageType =
  | 'status'
  | 'execution_start'
  | 'execution_cached'
  | 'executing'
  | 'progress'
  | 'executed'
  | 'execution_error'
  | 'execution_interrupted';

export interface ComfyWsStatus {
  type: 'status';
  data: { status: { exec_info: { queue_remaining: number } } };
}

export interface ComfyWsExecutionStart {
  type: 'execution_start';
  data: { prompt_id: string };
}

export interface ComfyWsExecuting {
  type: 'executing';
  data: { node: string | null; prompt_id: string };
}

export interface ComfyWsProgress {
  type: 'progress';
  data: { value: number; max: number; prompt_id: string; node: string };
}

export interface ComfyWsExecuted {
  type: 'executed';
  data: { node: string; output: Record<string, ComfyOutputFile[]>; prompt_id: string };
}

export interface ComfyWsError {
  type: 'execution_error';
  data: {
    prompt_id: string;
    node_id: string;
    node_type: string;
    exception_message: string;
    exception_type: string;
    traceback: string[];
  };
}

export type ComfyWsMessage =
  | ComfyWsStatus
  | ComfyWsExecutionStart
  | ComfyWsExecuting
  | ComfyWsProgress
  | ComfyWsExecuted
  | ComfyWsError
  | { type: string; data: unknown };

// ─── Connection state (used by useComfyConnection hook) ───────────────────────

export type ComfyConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface ComfyConnectionState {
  status: ComfyConnectionStatus;
  queueRemaining: number;
  currentPromptId: string | null;
  currentNode: string | null;
  progress: { value: number; max: number } | null;
}
