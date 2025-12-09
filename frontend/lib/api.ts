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
  if (API_BASE.startsWith('/')) {
    // Relative URL - construct from window.location
    if (typeof window !== 'undefined') {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${protocol}//${window.location.host}${path}`;
    }
    // Server-side fallback
    return `ws://localhost:8000${path}`;
  }
  // Absolute URL - replace http with ws
  return API_BASE.replace(/^http/, 'ws') + path.replace(/^\/api/, '');
};

// Helper for handling API responses
async function handleResponse(response: Response) {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
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
  /** Name of the training project */
  project_name: string;
  /** Model architecture type: SD1.5, SDXL, Flux, SD3, Lumina */
  model_type: string;
  /** Path to pretrained model checkpoint (.safetensors or .ckpt) */
  pretrained_model_name_or_path: string;
  /** Optional: Path to VAE model for encoding/decoding */
  vae_path?: string;

  /** Conditional: Path to CLIP-L model (required for Flux/SD3) */
  clip_l_path?: string;
  /** Conditional: Path to CLIP-G model (required for Flux/SD3) */
  clip_g_path?: string;
  /** Conditional: Path to T5-XXL model (required for Flux/SD3) */
  t5xxl_path?: string;

  /** Optional: Path to a LoRA model to continue training from */
  continue_from_lora?: string;
  /** Optional: Weights & Biases API key for logging */
  wandb_key?: string;

  // ========== DATASET & BASIC TRAINING ==========
  /** Directory containing training data */
  train_data_dir: string;
  /** Directory to save training outputs */
  output_dir: string;
  /** Resolution of training images (e.g., 1024 for SDXL) */
  resolution: number;
  /** Number of times to repeat the dataset. Higher = more training on same images. Typical: 10-20. */
  num_repeats: number;
  /** Maximum number of training epochs. */
  max_train_epochs: number;
  /** If > 0, overrides epochs. Leave at 0 to use epochs instead. */
  max_train_steps: number;
  /** Training batch size. */
  train_batch_size: number;
  /** Random seed for reproducibility. */
  seed: number;

  // Data augmentation
  /** Enable horizontal flipping for data augmentation */
  flip_aug: boolean;
  /** Enable random cropping for data augmentation */
  random_crop: boolean;
  /** Enable color augmentation */
  color_aug: boolean;
  /** Shuffle caption tokens during training */
  shuffle_caption: boolean;

  // ========== LEARNING RATES ==========
  /** Learning rate for the UNet model */
  unet_lr: number;
  /** Learning rate for the Text Encoder */
  text_encoder_lr: number;
  /** Learning rate scheduler type (e.g., "cosine", "linear") */
  lr_scheduler: string;
  /** Number of restarts for cosine_with_restarts, or degree for polynomial. */
  lr_scheduler_number: number;
  /** Ratio of total steps for learning rate warmup (0-1) */
  lr_warmup_ratio: number;
  /** If > 0, overrides warmup ratio. Leave at 0 to use ratio. */
  lr_warmup_steps: number;
  /** Power for polynomial scheduler. 1.0 = linear decay. */
  lr_power: number;

  // ========== LORA STRUCTURE ==========
  /** LoRA type (e.g., "LoRA", "LoCon", "LoHa") */
  lora_type: string;
  /** Network module to use (e.g., "networks.lora") */
  network_module: string;
  /** Dimension of the LoRA network (rank) */
  network_dim: number;
  /** Alpha parameter for LoRA (scaling) */
  network_alpha: number;
  /** Dimension for convolutional layers. Higher = better textures but slower. LoCon/LoHa only. */
  conv_dim: number;
  /** Alpha parameter for convolutional layers */
  conv_alpha: number;
  /** Dropout rate for the LoRA network (0-1) */
  network_dropout: number;
  /** Derive network_dim from weights */
  dim_from_weights: boolean;
  /** Decomposition factor for LoKR. -1 = auto-detect. Higher = more compact. */
  factor: number;
  /** Train normalization layers (LyCORIS) */
  train_norm: boolean;

  // Advanced LyCORIS parameters
  /** LyCORIS rank dropout (0-1) */
  rank_dropout: number;
  /** LyCORIS module dropout (0-1) */
  module_dropout: number;

  // Block-wise learning rates (advanced)
  /** e.g., "1,1,1,1,1,1,1,1,1,1,1,1" */
  down_lr_weight?: string;
  /** e.g., "1" */
  mid_lr_weight?: string;
  /** e.g., "1,1,1,1,1,1,1,1,1,1,1,1" */
  up_lr_weight?: string;
  /** e.g., "0.1" */
  block_lr_zero_threshold?: string;
  /** Per-block dimensions */
  block_dims?: string;
  /** Per-block alphas */
  block_alphas?: string;
  /** Per-block conv dimensions */
  conv_block_dims?: string;
  /** Per-block conv alphas */
  conv_block_alphas?: string;

  // ========== OPTIMIZER ==========
  /** Optimizer type (e.g., "AdamW8bit", "Lion8bit") */
  optimizer_type: string;
  /** Weight decay for optimizer */
  weight_decay: number;
  /** Gradient accumulation steps */
  gradient_accumulation_steps: number;
  /** Maximum gradient norm */
  max_grad_norm: number;
  /** JSON string of custom optimizer arguments. Advanced users only. */
  optimizer_args?: string;

  // ========== CAPTION & TOKEN CONTROL ==========
  /** Number of caption tokens to always keep (not subject to dropout). */
  keep_tokens: number;
  /** Skip last N CLIP layers. SD1.5: 1, SDXL: 2. */
  clip_skip: number;
  /** Maximum token length */
  max_token_length: number;
  /** Caption dropout rate (0-1) */
  caption_dropout_rate: number;
  /** Caption tag dropout rate (0-1) */
  caption_tag_dropout_rate: number;
  /** Caption dropout every N epochs */
  caption_dropout_every_n_epochs: number;
  /** Separator for keep_tokens */
  keep_tokens_separator: string;
  /** Secondary separator for captions */
  secondary_separator: string;
  /** Enable wildcard expansion in captions */
  enable_wildcard: boolean;
  /** Enable weighted captions */
  weighted_captions: boolean;

  // ========== BUCKETING ==========
  /** Enable bucketing for different resolutions */
  enable_bucket: boolean;
  /** Enable SDXL bucket optimization */
  sdxl_bucket_optimization: boolean;
  /** Minimum bucket resolution */
  min_bucket_reso: number;
  /** Maximum bucket resolution */
  max_bucket_reso: number;
  /** Do not upscale images in buckets */
  bucket_no_upscale: boolean;

  // ========== ADVANCED TRAINING ==========
  // SNR & Noise
  /** Enable min_snr_gamma */
  min_snr_gamma_enabled: boolean;
  /** Minimum SNR gamma value. Helps with convergence. Recommended: 5.0. */
  min_snr_gamma: number;
  /** Enable input perturbation noise */
  ip_noise_gamma_enabled: boolean;
  /** Input perturbation noise. Adds robustness. Typical: 0.05. */
  ip_noise_gamma: number;
  /** Enable multi-resolution noise */
  multinoise: boolean;
  /** Multi-resolution noise discount factor. Typical: 0.25. */
  multires_noise_discount: number;
  /** Noise offset */
  noise_offset: number;
  /** Adaptive noise scale */
  adaptive_noise_scale: number;
  /** Zero terminal SNR */
  zero_terminal_snr: boolean;

  // ========== MEMORY & PERFORMANCE ==========
  /** Enable gradient checkpointing */
  gradient_checkpointing: boolean;
  /** Mixed precision training (e.g., "fp16", "bf16") */
  mixed_precision: string;
  /** Use full FP16 training */
  full_fp16: boolean;
  /** Use FP8 for base model (experimental). Saves VRAM but may reduce quality. */
  fp8_base: boolean;
  /** VAE batch size */
  vae_batch_size: number;
  /** Do not use half VAE */
  no_half_vae: boolean;
  /** Cache latents */
  cache_latents: boolean;
  /** Cache latents to disk */
  cache_latents_to_disk: boolean;
  /** Cache text encoder outputs */
  cache_text_encoder_outputs: boolean;
  /** Cross attention mechanism (e.g., "sdpa", "xformers") */
  cross_attention: string;
  /** Number of persistent data loader workers (0=auto) */
  persistent_data_loader_workers: number;
  /** Disable token padding (memory optimization) */
  no_token_padding: boolean;

  // ========== SAVING & CHECKPOINTS ==========
  /** Save a checkpoint every N epochs */
  save_every_n_epochs: number;
  /** Save a checkpoint every N steps */
  save_every_n_steps: number;
  /** Save the last N epochs */
  save_last_n_epochs: number;
  /** Save the state of the last N epochs */
  save_last_n_epochs_state: number;
  /** Save training state for resuming */
  save_state: boolean;
  /** Save the state of the last N steps */
  save_last_n_steps_state: number;
  /** Format to save the model as (e.g., "safetensors", "ckpt") */
  save_model_as: string;
  /** Precision to save the model in (e.g., "fp16", "bf16") */
  save_precision: string;
  /** Output file name */
  output_name: string;
  /** Do not save metadata with the model */
  no_metadata: boolean;

  // ========== SAMPLE GENERATION ==========
  /** Generate samples every N epochs */
  sample_every_n_epochs: number;
  /** Generate samples every N steps */
  sample_every_n_steps: number;
  /** Optional: Path to a file containing sample prompts */
  sample_prompts?: string;
  /** Sampler to use for sample generation (e.g., "euler_a") */
  sample_sampler: string;

  // ========== LOGGING ==========
  /** Optional: Directory for logging */
  logging_dir?: string;
  /** Optional: Logging destination (e.g., "tensorboard", "wandb") */
  log_with?: string;
  /** Optional: Prefix for log files */
  log_prefix?: string;

  // ========== SD 2.x & ADVANCED ==========
  /** SD 2.x base model flag */
  v2: boolean;
  /** For SDXL v-pred or SD 2.x 768px */
  v_parameterization: boolean;
  /** Train U-Net only (recommended for SDXL) */
  network_train_unet_only: boolean;
  /** Prior loss weight */
  prior_loss_weight: number;

  // ========== FLUX-SPECIFIC PARAMETERS ==========
  /** Flux AutoEncoder path (*.safetensors) */
  ae_path?: string;
  /** Max tokens for T5-XXL (256 for schnell, 512 for dev) */
  t5xxl_max_token_length?: number;
  /** Apply attention mask to T5-XXL */
  apply_t5_attn_mask: boolean;
  /** Guidance scale for Flux.1 dev */
  guidance_scale: number;
  /** Timestep sampling (sigma, uniform, sigmoid, shift, flux_shift) */
  timestep_sampling: string;
  /** Scale for sigmoid timestep sampling */
  sigmoid_scale: number;
  /** Model prediction type (raw or additive for dev model) */
  model_prediction_type: string;
  /** Number of blocks to swap (memory optimization) */
  blocks_to_swap?: number;

  // ========== LUMINA-SPECIFIC PARAMETERS ==========
  /** Path to Gemma2 model (*.sft or *.safetensors), should be float16 */
  gemma2?: string;
  /** Maximum token length for Gemma2. Default: 256 */
  gemma2_max_token_length?: number;
  // ae_path, timestep_sampling, sigmoid_scale, blocks_to_swap are shared with Flux
}

export interface ValidationError {
  field: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
}

export interface TrainingStartResponse {
  success: boolean;
  message: string;
  training_id?: string;
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
