/**
 * Config Service - TOML configuration generation for Kohya training
 * Ported from Python services/trainers/kohya_toml.py
 *
 * CRITICAL: This file generates training configuration files.
 * Any bugs here can cause expensive VastAI training failures!
 *
 * Generates:
 * 1. dataset.toml - Images, folders, resolution, bucketing
 * 2. config.toml - Training args + Network args combined
 */

import fs from 'fs/promises';
import path from 'path';
import toml from '@iarna/toml';

// Import TrainingConfig type from frontend API definitions
// Note: This should match the Python TrainingConfig model
interface TrainingConfig {
  // Model and paths
  pretrained_model_name_or_path: string;
  train_data_dir: string;
  output_dir: string;
  output_name: string;
  model_type: 'SD15' | 'SDXL' | 'SD3' | 'FLUX' | 'LUMINA';
  vae_path?: string;
  continue_from_lora?: string;

  // Training parameters
  max_train_epochs: number;
  max_train_steps?: number;
  train_batch_size: number;
  resolution: number;
  seed: number;

  // Learning rate
  unet_lr: number;
  text_encoder_lr: number;
  lr_scheduler: string;
  lr_scheduler_number: number;
  lr_warmup_ratio: number;
  lr_warmup_steps: number;
  lr_power: number;

  // Optimizer
  optimizer_type: string;
  max_grad_norm: number;
  weight_decay: number;
  optimizer_args?: string[];

  // Network (LoRA) settings
  lora_type: string;
  network_dim: number;
  network_alpha: number;
  conv_dim?: number;
  conv_alpha?: number;
  network_dropout?: number;
  rank_dropout?: number;
  module_dropout?: number;
  train_norm?: boolean;
  factor?: number; // LoKR specific

  // Caption settings
  shuffle_caption: boolean;
  keep_tokens: number;
  max_token_length: number;
  clip_skip: number;
  weighted_captions: boolean;
  no_token_padding: boolean;
  caption_dropout_rate?: number;
  caption_tag_dropout_rate?: number;
  caption_dropout_every_n_epochs?: number;
  keep_tokens_separator?: string;
  secondary_separator?: string;
  enable_wildcard?: boolean;

  // Dataset settings
  num_repeats: number;
  min_bucket_reso: number;
  max_bucket_reso: number;
  bucket_reso_steps?: number;
  bucket_no_upscale?: boolean;

  // Augmentation
  flip_aug?: boolean;
  random_crop?: boolean;
  color_aug?: boolean;

  // Saving
  save_model_as: string;
  save_precision: string;
  no_metadata: boolean;
  save_every_n_epochs?: number;
  save_every_n_steps?: number;
  save_last_n_epochs?: number;
  save_last_n_epochs_state?: number;
  save_last_n_steps_state?: number;
  save_state?: boolean;
  save_state_on_train_end?: boolean;
  resume_from_state?: string;

  // Sampling
  sample_every_n_epochs?: number;
  sample_every_n_steps?: number;
  sample_sampler?: string;
  sample_prompts?: string;

  // Performance
  mixed_precision: string;
  gradient_checkpointing: boolean;
  gradient_accumulation_steps: number;
  cache_latents: boolean;
  cache_latents_to_disk: boolean;
  cache_text_encoder_outputs: boolean;
  cache_text_encoder_outputs_to_disk?: boolean;
  vae_batch_size: number;
  no_half_vae: boolean;
  persistent_data_loader_workers: boolean;
  max_data_loader_n_workers?: number;
  fp8_base: boolean;
  full_fp16: boolean;
  full_bf16?: boolean;
  lowram: boolean;
  cross_attention: string; // "xformers" | "sdpa"

  // Model-specific
  v2: boolean;
  v_parameterization: boolean;
  network_train_unet_only: boolean;

  // Noise settings
  noise_offset: number;
  zero_terminal_snr: boolean;
  prior_loss_weight: number;
  min_snr_gamma_enabled?: boolean;
  min_snr_gamma?: number;
  ip_noise_gamma_enabled?: boolean;
  ip_noise_gamma?: number;
  ip_noise_gamma_random_strength?: boolean;
  multinoise?: boolean;
  multires_noise_iterations?: number;
  multires_noise_discount?: number;
  adaptive_noise_scale?: number;

  // Timestep control
  min_timestep?: number;
  max_timestep?: number;

  // V-prediction loss variants
  scale_v_pred_loss_like_noise_pred?: boolean;
  v_pred_like_loss?: number;
  debiased_estimation_loss?: boolean;

  // Block-wise LR
  down_lr_weight?: string;
  mid_lr_weight?: string;
  up_lr_weight?: string;
  block_lr_zero_threshold?: number;
  block_dims?: string;
  block_alphas?: string;
  conv_block_dims?: string;
  conv_block_alphas?: string;

  // Logging
  logging_dir?: string;
  log_with?: string;
  log_prefix?: string;
  log_tracker_name?: string;
  log_tracker_config?: string;
  wandb_run_name?: string;

  // Flux-specific
  ae_path?: string;
  clip_l_path?: string;
  t5xxl_path?: string;
  t5xxl_max_token_length?: number;
  apply_t5_attn_mask?: boolean;
  guidance_scale?: number;
  timestep_sampling?: string;
  sigmoid_scale?: number;
  model_prediction_type?: string;
  blocks_to_swap?: number;

  // SD3-specific
  clip_g_path?: string;

  // Lumina-specific
  gemma2?: string;
  gemma2_max_token_length?: number;
}

// ========== TOML Generation Functions ==========

/**
 * Generate dataset.toml
 * This handles the [general] and [[datasets]] formatting required by Kohya
 */
export async function generateDatasetTOML(
  config: TrainingConfig,
  projectRoot: string,
  sdScriptsDir: string
): Promise<string> {
  // Validate paths first
  await validatePaths(config);

  const doc: any = {};

  // [general] section
  const general: any = {};

  // Resolution: For SDXL/SD3/Flux/Lumina, use list format [width, height]
  // For SD 1.5, single integer is acceptable
  if (['SDXL', 'SD3', 'FLUX', 'LUMINA'].includes(config.model_type)) {
    general.resolution = [config.resolution, config.resolution];
  } else {
    general.resolution = config.resolution;
  }

  general.shuffle_caption = config.shuffle_caption;

  doc.general = general;

  // [[datasets]] list
  const datasets: any[] = [];
  const dataset: any = {};

  if (config.keep_tokens > 0) {
    dataset.keep_tokens = config.keep_tokens;
  }

  // Resolution: Use same format as general section
  if (['SDXL', 'SD3', 'FLUX', 'LUMINA'].includes(config.model_type)) {
    dataset.resolution = [config.resolution, config.resolution];
  } else {
    dataset.resolution = config.resolution;
  }

  dataset.batch_size = config.train_batch_size;
  dataset.enable_bucket = true; // Force bucketing enabled
  dataset.min_bucket_reso = config.min_bucket_reso;
  dataset.max_bucket_reso = config.max_bucket_reso;

  if (config.bucket_reso_steps) {
    dataset.bucket_reso_steps = config.bucket_reso_steps;
  }
  if (config.bucket_no_upscale) {
    dataset.bucket_no_upscale = true;
  }

  // Augmentation settings
  if (config.flip_aug) {
    dataset.flip_aug = true;
  }
  if (config.random_crop) {
    dataset.random_crop = true;
  }
  if (config.color_aug) {
    dataset.color_aug = true;
  }

  // Caption settings
  if (config.caption_dropout_rate && config.caption_dropout_rate > 0) {
    dataset.caption_dropout_rate = config.caption_dropout_rate;
  }
  if (config.caption_tag_dropout_rate && config.caption_tag_dropout_rate > 0) {
    dataset.caption_tag_dropout_rate = config.caption_tag_dropout_rate;
  }
  if (config.caption_dropout_every_n_epochs && config.caption_dropout_every_n_epochs > 0) {
    dataset.caption_dropout_every_n_epochs = config.caption_dropout_every_n_epochs;
  }
  if (config.keep_tokens_separator && config.keep_tokens_separator !== '|||') {
    dataset.keep_tokens_separator = config.keep_tokens_separator;
  }
  if (config.secondary_separator) {
    dataset.secondary_separator = config.secondary_separator;
  }
  if (config.enable_wildcard) {
    dataset.enable_wildcard = true;
  }

  // [[datasets.subsets]] list
  const subsets: any[] = [];
  const subset: any = {};

  // Get absolute dataset path
  let datasetAbsPath = path.resolve(config.train_data_dir);

  // CRITICAL FIX: Use relative path from sd_scripts directory
  // Training scripts run from trainer/derrian_backend/sd_scripts/
  // Use POSIX paths for TOML compatibility (forward slashes)
  const relPath = path.relative(sdScriptsDir, datasetAbsPath);
  subset.image_dir = relPath.replace(/\\/g, '/'); // Convert Windows backslashes to forward slashes

  subset.num_repeats = config.num_repeats;

  subsets.push(subset);
  dataset.subsets = subsets;
  datasets.push(dataset);
  doc.datasets = datasets;

  return toml.stringify(doc as toml.JsonMap);
}

/**
 * Generate config.toml
 * Combines Training Arguments AND Network Arguments into one flat TOML
 */
export async function generateConfigTOML(
  config: TrainingConfig,
  projectRoot: string
): Promise<string> {
  // Validate paths first
  await validatePaths(config);

  // 1. Get base training args
  const args = getTrainingArguments(config);

  // 2. Add network args (LoRA settings) directly into same dict
  const networkConfig = getNetworkConfig(config);
  Object.assign(args, networkConfig);

  // Add manual network params
  args.network_dim = config.network_dim;
  args.network_alpha = config.network_alpha;

  if (config.conv_dim && config.conv_dim > 0) {
    args.conv_dim = config.conv_dim;
  }
  if (config.conv_alpha && config.conv_alpha > 0) {
    args.conv_alpha = config.conv_alpha;
  }

  if (config.network_dropout && config.network_dropout > 0) {
    args.network_dropout = config.network_dropout;
  }
  if (config.rank_dropout && config.rank_dropout > 0) {
    args.rank_dropout = config.rank_dropout;
  }
  if (config.module_dropout && config.module_dropout > 0) {
    args.module_dropout = config.module_dropout;
  }
  if (config.train_norm) {
    args.train_norm = true;
  }

  // LoKR specific
  if (config.lora_type === 'LoKR' && config.factor && config.factor !== -1) {
    args.factor = config.factor;
  }

  return toml.stringify(args as toml.JsonMap);
}

/**
 * Get network configuration based on LoRA type
 * Maps UI-friendly LoRA types to LyCORIS algorithm names
 */
function getNetworkConfig(config: TrainingConfig): any {
  const loraType = config.lora_type;

  // Standard LoRA variants
  if (loraType === 'LoRA') {
    return { network_module: 'networks.lora' };
  } else if (loraType === 'LoCon') {
    return { network_module: 'lycoris.kohya', network_args: ['algo=locon'] };
  } else if (loraType === 'LoHa') {
    return { network_module: 'lycoris.kohya', network_args: ['algo=loha'] };
  } else if (loraType === 'LoKR' || loraType === 'LoKr') {
    return { network_module: 'lycoris.kohya', network_args: ['algo=lokr'] };
  } else if (loraType === 'DoRA') {
    return {
      network_module: 'lycoris.kohya',
      network_args: ['algo=lora', 'weight_decompose=True'],
    };
  } else if (loraType === 'Full') {
    return { network_module: 'lycoris.kohya', network_args: ['algo=full'] };
  } else if (loraType === 'IA3') {
    return { network_module: 'lycoris.kohya', network_args: ['algo=ia3'] };
  } else if (loraType === 'DyLoRA') {
    return { network_module: 'lycoris.kohya', network_args: ['algo=dylora'] };
  } else if (loraType === 'GLoRA') {
    return { network_module: 'lycoris.kohya', network_args: ['algo=glora'] };
  } else if (loraType === 'Diag-OFT') {
    return { network_module: 'lycoris.kohya', network_args: ['algo=diag-oft'] };
  } else if (loraType === 'BOFT') {
    return { network_module: 'lycoris.kohya', network_args: ['algo=boft'] };
  }

  // Default fallback
  return { network_module: 'networks.lora' };
}

/**
 * Map TrainingConfig to Kohya CLI argument keys
 */
function getTrainingArguments(config: TrainingConfig): any {
  // Use POSIX paths (forward slashes) for all paths
  const args: any = {
    pretrained_model_name_or_path: path.resolve(config.pretrained_model_name_or_path).replace(/\\/g, '/'),
    max_train_epochs: config.max_train_epochs,
    output_dir: path.resolve(config.output_dir).replace(/\\/g, '/'),
    output_name: config.output_name,
    seed: config.seed,
    unet_lr: config.unet_lr,
    text_encoder_lr: config.text_encoder_lr,
    lr_scheduler: config.lr_scheduler,
    lr_scheduler_num_cycles: config.lr_scheduler_number,
    lr_warmup_ratio: config.lr_warmup_ratio,
    lr_warmup_steps: config.lr_warmup_steps,
    lr_power: config.lr_power,
    optimizer_type: config.optimizer_type,
    max_grad_norm: config.max_grad_norm,
    weight_decay: config.weight_decay,
    max_token_length: config.max_token_length,
    clip_skip: config.clip_skip,
    weighted_captions: config.weighted_captions,
    no_token_padding: config.no_token_padding,
    save_model_as: config.save_model_as,
    save_precision: config.save_precision,
    no_metadata: config.no_metadata,
    save_every_n_epochs: config.save_every_n_epochs || 0,
    save_every_n_steps: config.save_every_n_steps || 0,
    save_last_n_epochs: config.save_last_n_epochs || 0,
    save_last_n_epochs_state: config.save_last_n_epochs_state || 0,
    save_last_n_steps_state: config.save_last_n_steps_state || 0,
    save_state: config.save_state || false,
    sample_every_n_epochs: config.sample_every_n_epochs || 0,
    sample_every_n_steps: config.sample_every_n_steps || 0,
    sample_sampler: config.sample_sampler || 'euler_a',
    mixed_precision: config.mixed_precision,
    gradient_checkpointing: config.gradient_checkpointing,
    gradient_accumulation_steps: config.gradient_accumulation_steps,
    cache_latents: config.cache_latents,
    cache_latents_to_disk: config.cache_latents_to_disk,
    cache_text_encoder_outputs: config.cache_text_encoder_outputs,
    vae_batch_size: config.vae_batch_size,
    no_half_vae: config.no_half_vae,
    persistent_data_loader_workers: config.persistent_data_loader_workers ? 1 : 0,
    fp8_base: config.fp8_base,
    full_fp16: config.full_fp16,
    lowram: config.lowram,
    xformers: config.cross_attention === 'xformers',
    sdpa: config.cross_attention === 'sdpa',
    v2: config.v2,
    v_parameterization: config.v_parameterization,
    network_train_unet_only: config.network_train_unet_only,
    noise_offset: config.noise_offset,
    zero_terminal_snr: config.zero_terminal_snr,
    prior_loss_weight: config.prior_loss_weight,
  };

  // Performance/Memory
  if (config.cache_text_encoder_outputs_to_disk) {
    args.cache_text_encoder_outputs_to_disk = true;
  }
  if (config.full_bf16) {
    args.full_bf16 = true;
  }
  if (config.max_data_loader_n_workers) {
    args.max_data_loader_n_workers = config.max_data_loader_n_workers;
  }

  // Bucketing
  if (config.bucket_reso_steps) {
    args.bucket_reso_steps = config.bucket_reso_steps;
  }

  // Timestep control
  if (config.min_timestep !== undefined && config.min_timestep !== null) {
    args.min_timestep = config.min_timestep;
  }
  if (config.max_timestep !== undefined && config.max_timestep !== null) {
    args.max_timestep = config.max_timestep;
  }

  // V-prediction loss variants
  if (config.scale_v_pred_loss_like_noise_pred) {
    args.scale_v_pred_loss_like_noise_pred = true;
  }
  if (config.v_pred_like_loss !== undefined && config.v_pred_like_loss !== null) {
    args.v_pred_like_loss = config.v_pred_like_loss;
  }
  if (config.debiased_estimation_loss) {
    args.debiased_estimation_loss = true;
  }

  // State management
  if (config.save_state_on_train_end) {
    args.save_state_on_train_end = true;
  }
  if (config.resume_from_state) {
    args.resume = path.resolve(config.resume_from_state).replace(/\\/g, '/');
  }

  // Logging
  if (config.log_tracker_name) {
    args.log_tracker_name = config.log_tracker_name;
  }
  if (config.log_tracker_config) {
    args.log_tracker_config = config.log_tracker_config;
  }
  if (config.wandb_run_name) {
    args.wandb_run_name = config.wandb_run_name;
  }

  // Optional paths (use POSIX format)
  if (config.vae_path) {
    args.vae = path.resolve(config.vae_path).replace(/\\/g, '/');
  }
  if (config.continue_from_lora) {
    args.network_weights = path.resolve(config.continue_from_lora).replace(/\\/g, '/');
  }
  if (config.sample_prompts) {
    args.sample_prompts = path.resolve(config.sample_prompts).replace(/\\/g, '/');
  }
  if (config.logging_dir) {
    args.logging_dir = path.resolve(config.logging_dir).replace(/\\/g, '/');
  }
  if (config.log_with) {
    args.log_with = config.log_with;
  }
  if (config.log_prefix) {
    args.log_prefix = config.log_prefix;
  }
  if (config.optimizer_args) {
    args.optimizer_args = config.optimizer_args;
  }

  // Steps vs Epochs handling
  if (config.max_train_steps && config.max_train_steps > 0) {
    args.max_train_steps = config.max_train_steps;
  }

  // Noise Settings
  if (config.min_snr_gamma_enabled && config.min_snr_gamma) {
    args.min_snr_gamma = config.min_snr_gamma;
  }
  if (config.ip_noise_gamma_enabled && config.ip_noise_gamma) {
    args.ip_noise_gamma = config.ip_noise_gamma;
    if (config.ip_noise_gamma_random_strength) {
      args.ip_noise_gamma_random_strength = true;
    }
  }
  if (config.multinoise) {
    args.multires_noise_iterations = config.multires_noise_iterations || 6;
    args.multires_noise_discount = config.multires_noise_discount || 0.3;
  }
  if (config.adaptive_noise_scale && config.adaptive_noise_scale > 0) {
    args.adaptive_noise_scale = config.adaptive_noise_scale;
  }

  // Block-wise LR
  if (config.down_lr_weight) {
    args.down_lr_weight = config.down_lr_weight;
  }
  if (config.mid_lr_weight) {
    args.mid_lr_weight = config.mid_lr_weight;
  }
  if (config.up_lr_weight) {
    args.up_lr_weight = config.up_lr_weight;
  }
  if (config.block_lr_zero_threshold) {
    args.block_lr_zero_threshold = config.block_lr_zero_threshold;
  }
  if (config.block_dims) {
    args.block_dims = config.block_dims;
  }
  if (config.block_alphas) {
    args.block_alphas = config.block_alphas;
  }
  if (config.conv_block_dims) {
    args.conv_block_dims = config.conv_block_dims;
  }
  if (config.conv_block_alphas) {
    args.conv_block_alphas = config.conv_block_alphas;
  }

  // Flux specifics
  if (config.model_type === 'FLUX') {
    if (config.ae_path) {
      args.ae = path.resolve(config.ae_path).replace(/\\/g, '/');
    }
    if (config.clip_l_path) {
      args.clip_l = path.resolve(config.clip_l_path).replace(/\\/g, '/');
    }
    if (config.t5xxl_path) {
      args.t5xxl = path.resolve(config.t5xxl_path).replace(/\\/g, '/');
    }
    if (config.t5xxl_max_token_length) {
      args.t5xxl_max_token_length = config.t5xxl_max_token_length;
    }
    if (config.apply_t5_attn_mask) {
      args.apply_t5_attn_mask = true;
    }
    if (config.guidance_scale && config.guidance_scale !== 1.0) {
      args.guidance_scale = config.guidance_scale;
    }
    if (config.timestep_sampling) {
      args.timestep_sampling = config.timestep_sampling;
    }
    if (config.sigmoid_scale) {
      args.sigmoid_scale = config.sigmoid_scale;
    }
    if (config.model_prediction_type) {
      args.model_prediction_type = config.model_prediction_type;
    }
    if (config.blocks_to_swap) {
      args.blocks_to_swap = config.blocks_to_swap;
    }
  }

  // SD3 specifics
  if (config.model_type === 'SD3') {
    if (config.clip_l_path) {
      args.clip_l = path.resolve(config.clip_l_path).replace(/\\/g, '/');
    }
    if (config.clip_g_path) {
      args.clip_g = path.resolve(config.clip_g_path).replace(/\\/g, '/');
    }
    if (config.t5xxl_path) {
      args.t5xxl = path.resolve(config.t5xxl_path).replace(/\\/g, '/');
    }
  }

  // Lumina specifics
  if (config.model_type === 'LUMINA') {
    if (config.gemma2) {
      args.gemma2 = path.resolve(config.gemma2).replace(/\\/g, '/');
    }
    if (config.gemma2_max_token_length) {
      args.gemma2_max_token_length = config.gemma2_max_token_length;
    }
    if (config.ae_path) {
      args.ae = path.resolve(config.ae_path).replace(/\\/g, '/');
    }
  }

  return args;
}

/**
 * Validate critical paths exist before TOML generation
 */
async function validatePaths(config: TrainingConfig): Promise<void> {
  const errors: string[] = [];

  // Check base model
  const modelPath = path.resolve(config.pretrained_model_name_or_path);
  try {
    await fs.access(modelPath);
  } catch {
    errors.push(`Base model not found: ${modelPath}`);
  }

  // Check dataset directory
  const datasetPath = path.resolve(config.train_data_dir);
  try {
    await fs.access(datasetPath);
    const files = await fs.readdir(datasetPath);
    if (files.length === 0) {
      errors.push(`Training dataset directory is empty: ${datasetPath}`);
    }
  } catch {
    errors.push(`Training dataset directory not found: ${datasetPath}`);
  }

  // Check output directory parent exists
  const outputPath = path.resolve(config.output_dir);
  const outputParent = path.dirname(outputPath);
  try {
    await fs.access(outputParent);
  } catch {
    errors.push(`Output directory parent does not exist: ${outputParent}`);
  }

  // Optional paths (warn but don't fail)
  if (config.vae_path) {
    try {
      await fs.access(path.resolve(config.vae_path));
    } catch {
      console.warn(`VAE path specified but not found: ${config.vae_path}`);
    }
  }

  if (config.continue_from_lora) {
    try {
      await fs.access(path.resolve(config.continue_from_lora));
    } catch {
      errors.push(`LoRA to continue from not found: ${config.continue_from_lora}`);
    }
  }

  if (errors.length > 0) {
    const errorMsg = 'Path validation failed:\n' + errors.map(e => `  - ${e}`).join('\n');
    throw new Error(errorMsg);
  }
}

// ========== Main Service Class ==========

export class ConfigService {
  /**
   * Generate both dataset.toml and config.toml files
   */
  async generateTOMLs(
    config: TrainingConfig,
    outputDir: string
  ): Promise<{ dataset: string; config: string }> {
    // Project root is one level up from frontend directory
    const projectRoot = path.join(process.cwd(), '..');
    const sdScriptsDir = path.join(projectRoot, 'trainer', 'derrian_backend', 'sd_scripts');
    const runtimeStoreDir = path.join(projectRoot, 'trainer', 'runtime_store');

    // Ensure runtime store directory exists
    await fs.mkdir(runtimeStoreDir, { recursive: true });

    // Generate TOML content
    const datasetToml = await generateDatasetTOML(config, projectRoot, sdScriptsDir);
    const configToml = await generateConfigTOML(config, projectRoot);

    // Write files
    const datasetPath = path.join(runtimeStoreDir, 'dataset.toml');
    const configPath = path.join(runtimeStoreDir, 'config.toml');

    await fs.writeFile(datasetPath, datasetToml, 'utf-8');
    await fs.writeFile(configPath, configToml, 'utf-8');

    console.log('Generated dataset TOML:', datasetPath);
    console.log('Generated config TOML:', configPath);

    return {
      dataset: datasetPath,
      config: configPath,
    };
  }
}

// Export singleton instance
export const configService = new ConfigService();
