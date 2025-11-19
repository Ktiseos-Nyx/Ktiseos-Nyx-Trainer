/**
 * API client for backend communication
 * Centralized place for all API calls
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

// Helper for handling API responses
async function handleResponse(response: Response) {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
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
  list: async (path: string = '/workspace'): Promise<DirectoryListing> => {
    const response = await fetch(`${API_BASE}/files/list?path=${encodeURIComponent(path)}`);
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
};

// ========== Dataset Operations ==========

export interface ImageWithTags {
  image_path: string;
  image_name: string;
  tags: string[];
  has_tags: boolean;
}

export const datasetAPI = {
  list: async () => {
    const response = await fetch(`${API_BASE}/dataset/list`);
    return handleResponse(response);
  },

  uploadBatch: async (files: File[], datasetName: string) => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    const response = await fetch(
      `${API_BASE}/dataset/upload-batch?dataset_name=${encodeURIComponent(datasetName)}`,
      {
        method: 'POST',
        body: formData,
      }
    );
    return handleResponse(response);
  },

  tag: async (datasetPath: string, model: string = 'wd14-vit-v2', threshold: number = 0.35) => {
    const response = await fetch(`${API_BASE}/dataset/tag`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_path: datasetPath,
        model,
        threshold,
      }),
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
    const response = await fetch(`${API_BASE}/dataset/inject-trigger-word`, {
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
};

// ========== Training Operations ==========

export interface TrainingConfig {
  // ========== PROJECT & MODEL SETUP ==========
  project_name: string;
  model_type: string; // SD1.5, SDXL, Flux, SD3
  pretrained_model_name_or_path: string;
  vae_path?: string;

  // Conditional paths for Flux/SD3
  clip_l_path?: string;
  clip_g_path?: string;
  t5xxl_path?: string;

  continue_from_lora?: string;
  wandb_key?: string;

  // ========== DATASET & BASIC TRAINING ==========
  train_data_dir: string;
  output_dir: string;
  resolution: number;
  num_repeats: number; // CRITICAL for Kohya!
  max_train_epochs: number;
  max_train_steps: number; // 0 = use epochs instead
  train_batch_size: number;
  seed: number;

  // Data augmentation
  flip_aug: boolean;
  random_crop: boolean;
  color_aug: boolean;
  shuffle_caption: boolean;

  // ========== LEARNING RATES ==========
  unet_lr: number;
  text_encoder_lr: number;
  lr_scheduler: string;
  lr_scheduler_number: number;
  lr_warmup_ratio: number;
  lr_warmup_steps: number;
  lr_power: number;

  // ========== LORA STRUCTURE ==========
  lora_type: string;
  network_module: string;
  network_dim: number;
  network_alpha: number;
  conv_dim: number;
  conv_alpha: number;
  network_dropout: number;
  dim_from_weights: boolean;
  factor: number;
  train_norm: boolean;

  // Advanced LyCORIS parameters
  rank_dropout: number;
  module_dropout: number;

  // Block-wise learning rates (advanced)
  down_lr_weight?: string;
  mid_lr_weight?: string;
  up_lr_weight?: string;
  block_lr_zero_threshold?: string;
  block_dims?: string;
  block_alphas?: string;
  conv_block_dims?: string;
  conv_block_alphas?: string;

  // ========== OPTIMIZER ==========
  optimizer_type: string;
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
  sdxl_bucket_optimization: boolean;
  min_bucket_reso: number;
  max_bucket_reso: number;
  bucket_no_upscale: boolean;

  // ========== ADVANCED TRAINING ==========
  min_snr_gamma_enabled: boolean;
  min_snr_gamma: number;
  ip_noise_gamma_enabled: boolean;
  ip_noise_gamma: number;
  multinoise: boolean;
  multires_noise_discount: number;
  noise_offset: number;
  adaptive_noise_scale: number;
  zero_terminal_snr: boolean;

  // ========== MEMORY & PERFORMANCE ==========
  gradient_checkpointing: boolean;
  mixed_precision: string;
  full_fp16: boolean;
  fp8_base: boolean;
  vae_batch_size: number;
  no_half_vae: boolean;
  cache_latents: boolean;
  cache_latents_to_disk: boolean;
  cache_text_encoder_outputs: boolean;
  cross_attention: string;
  persistent_data_loader_workers: number;
  no_token_padding: boolean;

  // ========== SAVING & CHECKPOINTS ==========
  save_every_n_epochs: number;
  save_every_n_steps: number;
  save_last_n_epochs: number;
  save_last_n_epochs_state: number;
  save_state: boolean;
  save_last_n_steps_state: number;
  save_model_as: string;
  save_precision: string;
  output_name: string;
  no_metadata: boolean;

  // ========== SAMPLE GENERATION ==========
  sample_every_n_epochs: number;
  sample_every_n_steps: number;
  sample_prompts?: string;
  sample_sampler: string;

  // ========== LOGGING ==========
  logging_dir?: string;
  log_with?: string;
  log_prefix?: string;

  // ========== SD 2.x & ADVANCED ==========
  v2: boolean;
  v_parameterization: boolean;
  network_train_unet_only: boolean;
  prior_loss_weight: number;
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

  stop: async () => {
    const response = await fetch(`${API_BASE}/training/stop`, {
      method: 'POST',
    });
    return handleResponse(response);
  },

  status: async () => {
    const response = await fetch(`${API_BASE}/training/status`);
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
  connectLogs: (onMessage: (data: any) => void, onError?: (error: Event) => void) => {
    const wsUrl = `ws://localhost:8000/api/training/logs`;
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

  save: async (name: string, config: any) => {
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
  path: string;
  name: string;
  image_count: number;
  repeats: number;
  caption?: string;
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

  popular: async () => {
    const response = await fetch(`${API_BASE}/models/popular`);
    return handleResponse(response);
  },
};
