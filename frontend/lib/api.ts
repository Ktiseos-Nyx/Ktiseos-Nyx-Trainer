/**
 * API client for backend communication
 * Centralized place for all API calls
 *
 * Uses relative URLs by default (/api) so Next.js can proxy requests.
 * This works across local dev, VastAI, and Cloudflare tunnels.
 * Override with NEXT_PUBLIC_API_URL env var if needed.
 */

export const API_BASE = process.env.NEXT_PUBLIC_API_URL || '/api';

// Derive WebSocket base URL from API_BASE
// For relative URLs, construct WebSocket URL from current page location
// For absolute URLs, converts http:// -> ws:// and https:// -> wss://
export const getWsUrl = (path: string): string => {
  if (typeof window === 'undefined') return `ws://127.0.0.1:8000${path}`;

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  let host = window.location.host;

  // ✅ CASE 1: Standard Localhost or Vast Direct Mapping (3000 -> 8000)
  if (host.includes(':3000')) {
    host = host.replace(':3000', ':8000');
  }
  // ✅ CASE 2: Your Specific Vast Proxy Template (13000 -> 18000)
  else if (host.includes(':13000')) {
    host = host.replace(':13000', ':18000');
  }

  return `${protocol}//${host}${path}`;
};

export const WS_BASE = getWsUrl('');

// Helper for handling API responses
async function handleResponse(response: Response) {
  if (!response.ok) {
    // 1. Read the body as text FIRST (safely consumes the stream once)
    const text = await response.text();
    let errorMsg = `HTTP ${response.status}`;

    // 2. Try to parse that text as JSON to get a specific error message
    try {
      const json = JSON.parse(text);
      if (json.detail) {
        // Handle Pydantic validation errors (array of objects)
        if (typeof json.detail === 'object') {
          errorMsg = JSON.stringify(json.detail, null, 2);
        } else {
          errorMsg = String(json.detail);
        }
      }
    } catch {
      // 3. If it wasn't JSON, append the raw text preview
      if (text) {
        errorMsg += `: ${text.substring(0, 100)}`;
      }
    }

    throw new Error(errorMsg);
  }

  // Success handling
  const contentType = response.headers.get('content-type');
  if (contentType && contentType.includes('application/json')) {
    return response.json();
  }

  // Handle non-JSON success responses (rare but possible)
  const text = await response.text();
  try {
    return text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`Expected JSON response, got: ${text.substring(0, 100)}`);
  }
}

// ========== Type Definitions ==========

export interface WebSocketLogMessage {
  message?: string;
  log?: string;
  progress?: number;
  status?: string;
  type?: string;
  data?: number | string | object;
  [key: string]: unknown; // Allow additional properties
}

// ========== File Operations ==========

export interface FileInfo {
  name: string;
  path: string;
  type: 'file' | 'dir';
  size: number;
  modified: number;
  is_image?: boolean;
  mime_type?: string;
}

export interface DirectoryListing {
  path: string;
  parent: string | null;
  files: FileInfo[];
}

// Add these helper types at the top of lib/api.ts
export type ModelType = 'SD1.5' | 'SDXL' | 'Flux' | 'SD3' | 'SD3.5' | 'Lumina' | 'Chroma';
export type LoRAType = 'LoRA' | 'LoCon' | 'LoHa' | 'LoKr' | 'DoRA';
export type OptimizerType = 'AdamW' | 'AdamW8bit' | 'Lion' | 'Lion8bit' | 'SGDNesterov' | 'SGDNesterov8bit' | 'DAdaptation' | 'DAdaptAdam' | 'DAdaptAdaGrad' | 'DAdaptAdan' | 'DAdaptSGD' | 'Prodigy' | 'AdaFactor' | 'CAME';
export type SchedulerType = 'linear' | 'cosine' | 'cosine_with_restarts' | 'polynomial' | 'constant' | 'constant_with_warmup' | 'adafactor';

export const fileAPI = {
  getDefaultWorkspace: async (): Promise<{ path: string; allowed_dirs: string[] }> => {
    const response = await fetch(`${API_BASE}/files/default-workspace`);
    return handleResponse(response);
  },

  list: async (path?: string): Promise<DirectoryListing> => {
    const url = path
      ? `${API_BASE}/files/list?path=${encodeURIComponent(path)}`
      : `${API_BASE}/files/list`;
    const response = await fetch(url);
    return handleResponse(response);
  },

  upload: async (file: File, destination: string) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(
      `${API_BASE}/files/upload?destination=${encodeURIComponent(destination)}`,
      {
        method: 'POST',
        body: formData,
      }
    );
    return handleResponse(response);
  },

  delete: async (path: string) => {
    const response = await fetch(
      `${API_BASE}/files/delete?path=${encodeURIComponent(path)}`,
      { method: 'DELETE' }
    );
    return handleResponse(response);
  },

  rename: async (oldPath: string, newName: string) => {
    const response = await fetch(`${API_BASE}/files/rename`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ old_path: oldPath, new_name: newName }),
    });
    return handleResponse(response);
  },

  mkdir: async (path: string, name: string) => {
    const response = await fetch(`${API_BASE}/files/mkdir`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, name }),
    });
    return handleResponse(response);
  },

  read: async (path: string) => {
    const response = await fetch(`${API_BASE}/files/read/${encodeURIComponent(path.substring(1))}`);
    return handleResponse(response);
  },

  write: async (path: string, content: string) => {
    const response = await fetch(`${API_BASE}/files/write`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, content }),
    });
    return handleResponse(response);
  },

  // Get image URL for displaying thumbnails/previews
  getImageUrl: (path: string) => {
    // Remove leading slash if present
    const cleanPath = path.startsWith('/') ? path.substring(1) : path;
    return `${API_BASE}/files/image/${cleanPath}`;
  },
};

// ========== Dataset Operations ==========

export interface ImageWithTags {
  image_path: string;
  image_name: string;
  tags: string[];
  has_tags: boolean;
  url?: string;
}

export interface PaginatedDatasetsResponse {
  datasets: DatasetInfo[];
  total: number;
  pages: number;
  page: number;
}

export const datasetAPI = {
  // ✅ Now uses Python backend
  list: async (): Promise<{ datasets: DatasetInfo[]; total: number }> => {
    const response = await fetch(`${API_BASE}/dataset/list`);
    return handleResponse(response);
  },

  // ✅ New method
  deleteDataset: async (name: string) => {
    const response = await fetch(`${API_BASE}/dataset/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    });
    return handleResponse(response);
  },

  tag: async (params: {
    datasetDir: string;
    model?: string;
    forceDownload?: boolean;
    threshold?: number;
    generalThreshold?: number | null;
    characterThreshold?: number | null;
    captionExtension?: string;
    captionSeparator?: string;
    undesiredTags?: string;
    tagReplacement?: string | null;
    alwaysFirstTags?: string | null;
    characterTagsFirst?: boolean;
    useRatingTags?: boolean;
    useRatingTagsAsLastTag?: boolean;
    removeUnderscore?: boolean;
    characterTagExpand?: boolean;
    appendTags?: boolean;
    recursive?: boolean;
    batchSize?: number;
    maxWorkers?: number;
    useOnnx?: boolean;
    frequencyTags?: boolean;
    debug?: boolean;
  }) => {
    const response = await fetch(`${API_BASE}/dataset/tag`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_dir: params.datasetDir,
        model: params.model ?? 'SmilingWolf/wd-vit-large-tagger-v3',
        force_download: params.forceDownload ?? false,
        threshold: params.threshold ?? 0.35,
        general_threshold: params.generalThreshold ?? null,
        character_threshold: params.characterThreshold ?? null,
        caption_extension: params.captionExtension ?? '.txt',
        caption_separator: params.captionSeparator ?? ', ',
        undesired_tags: params.undesiredTags ?? '',
        tag_replacement: params.tagReplacement ?? null,
        always_first_tags: params.alwaysFirstTags ?? null,
        character_tags_first: params.characterTagsFirst ?? false,
        use_rating_tags: params.useRatingTags ?? false,
        use_rating_tags_as_last_tag: params.useRatingTagsAsLastTag ?? false,
        remove_underscore: params.removeUnderscore ?? true,
        character_tag_expand: params.characterTagExpand ?? false,
        append_tags: params.appendTags ?? false,
        recursive: params.recursive ?? false,
        batch_size: params.batchSize ?? 8,
        max_workers: params.maxWorkers ?? 2,
        use_onnx: params.useOnnx ?? true,
        frequency_tags: params.frequencyTags ?? false,
        debug: params.debug ?? false,
      }),
    });
    return handleResponse(response);
  },

  getTaggingStatus: async (jobId: string) => {
    const response = await fetch(`${API_BASE}/dataset/tag/status/${jobId}`);
    return handleResponse(response);
  },

  stopTagging: async (jobId: string) => {
    const response = await fetch(`${API_BASE}/dataset/tag/stop/${jobId}`, {
      method: 'POST',
    });
    return handleResponse(response);
  },

  create: async (name: string) => {
    const response = await fetch(`${API_BASE}/dataset/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    return handleResponse(response);
  },

  // Tag Management
  getImagesWithTags: async (datasetPath: string): Promise<{ images: ImageWithTags[], total: number }> => {
    const response = await fetch(`${API_BASE}/dataset/images-with-tags?dataset_path=${encodeURIComponent(datasetPath)}`);
    return handleResponse(response);
  },

  updateTags: async (imagePath: string, tags: string[]) => {
    const response = await fetch(`${API_BASE}/dataset/update-tags`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_path: imagePath, tags }),
    });
    return handleResponse(response);
  },

  bulkTagOperation: async (datasetPath: string, operation: 'add' | 'remove' | 'replace', tags: string[], replaceWith?: string) => {
    const response = await fetch(`${API_BASE}/dataset/bulk-tag-operation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_path: datasetPath,
        operation,
        tags,
        replace_with: replaceWith || ''
      }),
    });
    return handleResponse(response);
  },

  injectTriggerWord: async (datasetPath: string, triggerWord: string, position: 'start' | 'end' = 'start') => {
    const response = await fetch(`${API_BASE}/dataset/captions/add-trigger`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_path: datasetPath,
        trigger_word: triggerWord,
        position
      }),
    });
    return handleResponse(response);
  },

  // Upload multiple files in one request
  uploadBatch: async (files: File[], datasetName: string) => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    formData.append('dataset_name', datasetName);

    const response = await fetch(`${API_BASE}/dataset/upload-batch`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse(response);
  },

  // WebSocket for tagging logs (job-based)
  connectTaggingLogs: (jobId: string, onMessage: (data: WebSocketLogMessage) => void, onError?: (error: Event) => void) => {
    const wsUrl = `${WS_BASE}/ws/jobs/${jobId}/logs`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };

    if (onError) {
      ws.onerror = onError;
    }

    return ws;
  },
};

// ========== Captioning Operations (BLIP/GIT) ==========

export interface BLIPConfig {
  dataset_dir: string;
  caption_extension?: string;
  caption_weights?: string;
  batch_size?: number;
  max_workers?: number;
  beam_search?: boolean;
  num_beams?: number;
  top_p?: number;
  max_length?: number;
  min_length?: number;
  recursive?: boolean;
  debug?: boolean;
}

export interface GITConfig {
  dataset_dir: string;
  caption_extension?: string;
  model_id?: string;
  batch_size?: number;
  max_workers?: number;
  max_length?: number;
  remove_words?: boolean;
  recursive?: boolean;
  debug?: boolean;
}

export const captioningAPI = {
  startBLIP: async (config: BLIPConfig) => {
    const response = await fetch(`${API_BASE}/dataset/caption/blip`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    return handleResponse(response);
  },

  startGIT: async (config: GITConfig) => {
    const response = await fetch(`${API_BASE}/dataset/caption/git`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    return handleResponse(response);
  },

  getStatus: async (jobId: string) => {
    const response = await fetch(`${API_BASE}/dataset/caption/status/${jobId}`);
    return handleResponse(response);
  },

  stop: async (jobId: string) => {
    const response = await fetch(`${API_BASE}/dataset/caption/stop/${jobId}`, {
      method: 'POST',
    });
    return handleResponse(response);
  },

  // WebSocket for captioning logs (reuses same endpoint as tagging)
  connectLogs: (jobId: string, onMessage: (data: WebSocketLogMessage) => void, onError?: (error: Event) => void) => {
    const wsUrl = `${WS_BASE}/ws/jobs/${jobId}/logs`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };

    if (onError) {
      ws.onerror = onError;
    }

    return ws;
  },
};

// ========== Training Operations ==========

export interface TrainingConfig {
  // ========== PROJECT & MODEL SETUP ==========
  project_name: string;
  // ✅ FIX: Match the Zod Enum exactly
  model_type: 'SD1.5' | 'SDXL' | 'Flux' | 'SD3' | 'SD3.5' | 'Lumina' | 'Chroma';
  pretrained_model_name_or_path: string;
  vae_path?: string;
  ae_path?: string;
  clip_l_path?: string;
  clip_g_path?: string;
  t5xxl_path?: string;
  continue_from_lora?: string;
  wandb_key?: string;

  // ========== DATASET & BASIC TRAINING ==========
  train_data_dir: string;
  output_dir: string;
  resolution: number;
  num_repeats: number;
  max_train_epochs: number;
  max_train_steps: number;
  train_batch_size: number;
  seed: number;
  flip_aug: boolean;
  random_crop: boolean;
  color_aug: boolean;
  shuffle_caption: boolean;

  // ========== LEARNING RATES ==========
  unet_lr: number;
  text_encoder_lr: number;
  lr_scheduler: SchedulerType; // ✅ Strict
  lr_scheduler_number: number;
  lr_warmup_ratio: number;
  lr_warmup_steps: number;
  lr_power: number;

  // ========== LORA STRUCTURE ==========
  lora_type: LoRAType; // ✅ Strict
  network_module: string;
  network_dim: number;
  network_alpha: number;
  conv_dim: number;
  conv_alpha: number;
  network_dropout: number;
  dim_from_weights: boolean;
  factor: number;
  train_norm: boolean;
  rank_dropout: number;
  module_dropout: number;

  // ========== OPTIMIZER ==========
  optimizer_type: OptimizerType; // ✅ Strict (FIXES ERROR 2322)
  weight_decay: number;
  gradient_accumulation_steps: number;
  max_grad_norm: number;
  optimizer_args?: string;

  // ========== CAPTION & TOKEN CONTROL ==========
  keep_tokens: number;
  clip_skip: number;
  max_token_length: number;
  caption_dropout_rate: number;
  caption_tag_dropout_rate: number;
  caption_dropout_every_n_epochs: number;
  keep_tokens_separator: string;
  secondary_separator: string;
  enable_wildcard: boolean;
  weighted_captions: boolean;

  // ========== BUCKETING ==========
  enable_bucket: boolean;
  sdxl_bucket_optimization?: boolean;
  min_bucket_reso: number;
  max_bucket_reso: number;
  bucket_reso_steps?: number;
  bucket_no_upscale: boolean;

  // ========== MEMORY & PERFORMANCE ==========
  gradient_checkpointing?: boolean;
  mixed_precision: 'no' | 'fp16' | 'bf16';
  full_fp16: boolean;
  full_bf16?: boolean;
  fp8_base: boolean;
  vae_batch_size: number;
  no_half_vae: boolean;
  cache_latents: boolean;
  cache_latents_to_disk: boolean;
  cache_text_encoder_outputs: boolean;
  cache_text_encoder_outputs_to_disk?: boolean;
  cross_attention?: string;
  persistent_data_loader_workers: number;
  no_token_padding?: boolean;

  // ========== SAVING & CHECKPOINTS ==========
  save_every_n_epochs: number;
  save_every_n_steps: number;
  save_last_n_epochs: number;
  save_last_n_epochs_state: number;
  save_state: boolean;
  save_state_on_train_end?: boolean;
  save_last_n_steps_state?: number;
  save_model_as: 'safetensors' | 'ckpt' | 'pt';
  save_precision: string;
  output_name?: string;
  no_metadata?: boolean;
  resume_from_state?: string;

  // ========== SAMPLE GENERATION ==========
  sample_every_n_epochs: number;
  sample_every_n_steps: number;
  sample_prompts?: string;
  sample_sampler: string;

  // ========== ADVANCED NOISE / SNR ==========
  min_snr_gamma_enabled?: boolean;
  min_snr_gamma: number;
  ip_noise_gamma_enabled?: boolean;
  ip_noise_gamma: number;
  multinoise?: boolean;
  multires_noise_discount: number;
  noise_offset: number;
  adaptive_noise_scale: number;
  zero_terminal_snr?: boolean;

  // ========== ADDITIONAL ADVANCED ==========
  scale_v_pred_loss_like_noise_pred?: boolean;
  v_pred_like_loss?: number;
  debiased_estimation_loss?: boolean;
  lowram?: boolean;
  max_data_loader_n_workers?: number;
  min_timestep?: number;
  max_timestep?: number;
  multires_noise_iterations?: number;
  ip_noise_gamma_random_strength?: boolean;

  // ========== FLUX-SPECIFIC ==========
  t5xxl_max_token_length?: number;
  apply_t5_attn_mask?: boolean;
  guidance_scale?: number; // ✅ FIXED: Now TypeScript knows this exists!
  timestep_sampling?: string;
  sigmoid_scale?: number;
  model_prediction_type?: string;
  blocks_to_swap?: number;

  // ========== LUMINA-SPECIFIC ==========
  gemma2?: string;
  gemma2_max_token_length?: number;

  // ========== LOGGING ==========
  logging_dir?: string;
  log_with?: string;
  log_prefix?: string;
  log_tracker_name?: string;
  log_tracker_config?: string;
  wandb_run_name?: string;
}

export interface ValidationError {
  field: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
}

export interface TrainingStartResponse {
  success: bool;
  message: string;
  job_id?: string;
  warnings?: string[];  // Deprecated
  validation_errors?: ValidationError[];
}

export const trainingAPI = {
  start: async (config: TrainingConfig) => {
    const response = await fetch(`${API_BASE}/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    return handleResponse(response);
  },

  stop: async (jobId: string) => {
    const response = await fetch(`${API_BASE}/training/stop/${jobId}`, {
      method: 'POST',
    });
    return handleResponse(response);
  },

  status: async (jobId: string) => {
    const response = await fetch(`${API_BASE}/training/status/${jobId}`);
    return handleResponse(response);
  },

  validate: async (config: TrainingConfig) => {
    const response = await fetch(`${API_BASE}/training/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    return handleResponse(response);
  },

  // WebSocket for logs
  connectLogs: (jobId: string, onMessage: (data: WebSocketLogMessage) => void, onError?: (error: Event) => void) => {
    const wsUrl = `${WS_BASE}/ws/jobs/${jobId}/logs`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };

    if (onError) {
      ws.onerror = onError;
    }

    return ws;
  },
};

// ========== Config Operations ==========

export const configAPI = {
  templates: async () => {
    const response = await fetch(`${API_BASE}/config/templates`);
    return handleResponse(response);
  },

  load: async (path: string) => {
    const response = await fetch(`${API_BASE}/config/load?path=${encodeURIComponent(path)}`);
    return handleResponse(response);
  },

  save: async (name: string, config: Record<string, unknown>) => {
    const response = await fetch(`${API_BASE}/config/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, config }),
    });
    return handleResponse(response);
  },

  defaults: async () => {
    const response = await fetch(`${API_BASE}/config/defaults`);
    return handleResponse(response);
  },
};

// ========== Utilities Operations ==========

export interface CalculatorRequest {
  dataset_path: string;
  epochs: number;
  batch_size: number;
}

export interface CalculatorResponse {
  success: boolean;
  dataset_path: string;
  images: number;
  repeats: number;
  epochs: number;
  batch_size: number;
  total_steps: number;
  caption?: string;
  time_estimate_min: number;
  time_estimate_max: number;
  recommendation: string;
}

export interface DatasetInfo {
  name: string;
  path: string;
  image_count: number;
  caption_count: number;
  total_size: number;
  created_at?: string;
  modified_at?: string;
  tags_present?: boolean;
}

export interface LoRAFile {
  name: string;
  path: string;
  size: number;
  size_formatted: string;
  modified: number;
}

export interface HuggingFaceUploadRequest {
  hf_token: string;
  owner: string;
  repo_name: string;
  repo_type: string;
  selected_files: string[];
  remote_folder?: string;
  commit_message?: string;
  create_pr?: boolean;
}

export const utilitiesAPI = {
  // Directories
  getDirectories: async () => {
    const response = await fetch(`${API_BASE}/utilities/directories`);
    return handleResponse(response);
  },

  // Calculator
  calculateSteps: async (request: CalculatorRequest): Promise<CalculatorResponse> => {
    const response = await fetch(`${API_BASE}/utilities/calculator`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse(response);
  },

  browseDatasets: async (): Promise<{ datasets: DatasetInfo[] }> => {
    const response = await fetch(`${API_BASE}/utilities/datasets/browse`);
    return handleResponse(response);
  },

  // LoRA File Management
  listLoraFiles: async (directory: string, extension: string = 'safetensors', sortBy: string = 'name') => {
    const response = await fetch(`${API_BASE}/utilities/lora/list`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ directory, file_extension: extension, sort_by: sortBy }),
    });
    return handleResponse(response);
  },

  getResizeDimensions: async () => {
    const response = await fetch(`${API_BASE}/utilities/lora/resize-dimensions`);
    return handleResponse(response);
  },

  resizeLora: async (inputPath: string, outputPath: string, newDim: number, newAlpha: number) => {
    const response = await fetch(`${API_BASE}/utilities/lora/resize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        input_path: inputPath,
        output_path: outputPath,
        new_dim: newDim,
        new_alpha: newAlpha
      }),
    });
    return handleResponse(response);
  },

  mergeLora: async (
    loraInputs: Array<{ path: string; ratio: number }>,
    outputPath: string,
    modelType: string = 'sd',
    device: string = 'cpu',
    savePrecision: string = 'fp16',
    precision: string = 'float'
  ) => {
    const response = await fetch(`${API_BASE}/utilities/lora/merge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        lora_inputs: loraInputs,
        output_path: outputPath,
        model_type: modelType,
        device,
        save_precision: savePrecision,
        precision,
      }),
    });
    return handleResponse(response);
  },

  // HuggingFace
  uploadToHuggingFace: async (request: HuggingFaceUploadRequest) => {
    const response = await fetch(`${API_BASE}/utilities/hf/upload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse(response);
  },

  validateHfToken: async (token: string) => {
    const response = await fetch(`${API_BASE}/utilities/hf/validate-token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hf_token: token }),
    });
    return handleResponse(response);
  },
};

// ========== Models & VAEs ==========

export interface ModelFile {
  name: string;
  path: string;
  size_mb: number;
  type: 'model' | 'vae';
}

export interface PopularModel {
  name: string;
  url: string;
  description: string;
}

export const modelsAPI = {
  download: async (url: string, downloadType: 'model' | 'vae', modelType?: string) => {
    const response = await fetch(`${API_BASE}/models/download`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url,
        download_type: downloadType,
        model_type: modelType,
      }),
    });
    return handleResponse(response);
  },

  list: async () => {
    const response = await fetch(`${API_BASE}/models/list`);
    return handleResponse(response);
  },

  delete: async (fileType: 'model' | 'vae', fileName: string) => {
    const response = await fetch(
      `${API_BASE}/models/${fileType}/${encodeURIComponent(fileName)}`,
      { method: 'DELETE' }
    );
    return handleResponse(response);
  },

  cancel: async () => {
    const response = await fetch(`${API_BASE}/models/cancel`, {
      method: 'POST',
    });
    return handleResponse(response);
  },

  popular: async () => {
    const response = await fetch(`${API_BASE}/models/popular`);
    return handleResponse(response);
  },
};

// ========== Civitai Browse ==========
// Attribution: Inspired by sd-webui-civbrowser
// https://github.com/SignalFlagZ/sd-webui-civbrowser

export interface CivitaiImage {
  url: string;
  nsfw: boolean;
  width: number;
  height: number;
  hash: string;
  meta?: Record<string, unknown>;
}

export interface CivitaiModelVersion {
  id: number;
  modelId: number;
  name: string;
  description: string;
  downloadUrl: string;
  trainedWords: string[];
  images: CivitaiImage[];
  files: Array<{
    name: string;
    id: number;
    sizeKB: number;
    type: string;
    downloadUrl: string;
  }>;
}

export interface CivitaiModel {
  id: number;
  name: string;
  description: string;
  type: string;
  nsfw: boolean;
  tags: string[];
  creator: {
    username: string;
    image?: string;
  };
  stats: {
    downloadCount: number;
    favoriteCount: number;
    commentCount: number;
    ratingCount: number;
    rating: number;
  };
  modelVersions: CivitaiModelVersion[];
}

export interface CivitaiBrowseParams {
  limit?: number;
  page?: number;
  cursor?: string;
  query?: string;
  tag?: string;
  username?: string;
  types?: string;
  baseModel?: string;
  sort?: string;
  period?: string;
  nsfw?: boolean;
}

export interface CivitaiTag {
  name: string;
  modelCount: number;
  color?: string;
}

export const civitaiAPI = {
  browse: async (params: CivitaiBrowseParams = {}) => {
    const queryParams = new URLSearchParams();

    if (params.limit) queryParams.append('limit', params.limit.toString());
    if (params.cursor) queryParams.append('cursor', params.cursor);
    if (params.page) queryParams.append('page', params.page.toString());
    if (params.query) queryParams.append('query', params.query);
    if (params.tag) queryParams.append('tag', params.tag);
    if (params.username) queryParams.append('username', params.username);
    if (params.types) queryParams.append('types', params.types);
    if (params.baseModel) queryParams.append('baseModel', params.baseModel);
    if (params.sort) queryParams.append('sort', params.sort);
    if (params.period) queryParams.append('period', params.period);
    if (params.nsfw !== undefined) queryParams.append('nsfw', params.nsfw.toString());

    const response = await fetch(`${API_BASE}/civitai/models?${queryParams}`);
    return handleResponse(response);
  },

  getModel: async (modelId: number) => {
    const response = await fetch(`${API_BASE}/civitai/models/${modelId}`);
    return handleResponse(response);
  },

  getModelVersion: async (versionId: number) => {
    const response = await fetch(`${API_BASE}/civitai/model-versions/${versionId}`);
    return handleResponse(response);
  },

  getTags: async (limit: number = 20, query?: string) => {
    const queryParams = new URLSearchParams();
    queryParams.append('limit', limit.toString());
    if (query) queryParams.append('query', query);

    const response = await fetch(`${API_BASE}/civitai/tags?${queryParams}`);
    return handleResponse(response);
  },

  download: async (
    modelId: number,
    versionId: number,
    downloadUrl: string,
    filename: string,
    modelType: 'model' | 'vae' = 'model'
  ) => {
    const response = await fetch(`${API_BASE}/civitai/download`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_id: modelId,
        version_id: versionId,
        download_url: downloadUrl,
        filename,
        model_type: modelType,
      }),
    });
    return handleResponse(response);
  },
};

// ========== Caption Editing API ==========

export const captionAPI = {
  addTrigger: async (params: {
    dataset_path: string;
    trigger_word: string;
    position?: 'first' | 'last';
  }) => {
    const response = await fetch(`${API_BASE}/dataset/captions/add-trigger`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_path: params.dataset_path,
        trigger_word: params.trigger_word,
        position: params.position || 'first',
      }),
    });
    return handleResponse(response);
  },

  removeTags: async (params: {
    dataset_path: string;
    tags_to_remove: string[];
  }) => {
    const response = await fetch(`${API_BASE}/dataset/captions/remove-tags`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_path: params.dataset_path,
        tags_to_remove: params.tags_to_remove,
      }),
    });
    return handleResponse(response);
  },

  replace: async (params: {
    dataset_path: string;
    find: string;
    replace: string;
    use_regex?: boolean;
  }) => {
    const response = await fetch(`${API_BASE}/dataset/captions/replace`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_path: params.dataset_path,
        find: params.find,
        replace: params.replace,
        use_regex: params.use_regex || false,
      }),
    });
    return handleResponse(response);
  },
};
