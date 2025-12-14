/**
 * Zod validation schemas for training configuration
 * Provides type-safe validation and helpful error messages
 */

import { z } from 'zod';

/**
 * Model type enumeration
 */
export const ModelTypeSchema = z.enum(['SD1.5', 'SDXL', 'Flux', 'SD3', 'SD3.5', 'Lumina', 'Chroma'], {
  errorMap: () => ({ message: 'Please select a valid model type' }),
});

/**
 * LoRA type enumeration
 */
export const LoRATypeSchema = z.enum(['LoRA', 'LoCon', 'LoHa', 'LoKr', 'DoRA'], {
  errorMap: () => ({ message: 'Please select a valid LoRA type' }),
});

/**
 * Optimizer enumeration
 */
export const OptimizerSchema = z.enum([
  'AdamW',
  'AdamW8bit',
  'Lion',
  'Lion8bit',
  'SGDNesterov',
  'SGDNesterov8bit',
  'DAdaptation',
  'DAdaptAdam',
  'DAdaptAdaGrad',
  'DAdaptAdan',
  'DAdaptSGD',
  'Prodigy',
  'AdaFactor',
  'CAME',
], {
  errorMap: () => ({ message: 'Please select a valid optimizer' }),
});

/**
 * Learning rate scheduler enumeration
 */
export const LRSchedulerSchema = z.enum([
  'linear',
  'cosine',
  'cosine_with_restarts',
  'polynomial',
  'constant',
  'constant_with_warmup',
  'adafactor',
], {
  errorMap: () => ({ message: 'Please select a valid learning rate scheduler' }),
});

/**
 * Mixed precision enumeration
 */
export const MixedPrecisionSchema = z.enum(['no', 'fp16', 'bf16'], {
  errorMap: () => ({ message: 'Please select a valid precision mode' }),
});

/**
 * Save format enumeration
 */
export const SaveFormatSchema = z.enum(['safetensors', 'ckpt', 'pt'], {
  errorMap: () => ({ message: 'Please select a valid save format' }),
});

/**
 * Training configuration validation schema
 * Matches the TrainingConfig interface with runtime validation
 */
export const TrainingConfigSchema = z.object({
  // ========== PROJECT & MODEL SETUP ==========
  project_name: z
    .string()
    .min(1, 'Project name is required')
    .max(100, 'Project name must be less than 100 characters')
    .regex(/^[a-zA-Z0-9_-]+$/, 'Project name must contain only letters, numbers, underscores, and hyphens'),

  model_type: ModelTypeSchema,

  pretrained_model_name_or_path: z
    .string()
    .min(1, 'Pretrained model path is required')
    .refine((val) => val.endsWith('.safetensors') || val.endsWith('.ckpt') || val.includes('huggingface'), {
      message: 'Model path must be a .safetensors or .ckpt file, or a HuggingFace model ID',
    }),

  vae_path: z.string().optional(),
  clip_l_path: z.string().optional(),
  clip_g_path: z.string().optional(),
  t5xxl_path: z.string().optional(),
  continue_from_lora: z.string().optional(),
  wandb_key: z.string().optional(),

  // ========== DATASET & BASIC TRAINING ==========
  train_data_dir: z.string().min(1, 'Training data directory is required'),
  output_dir: z.string().min(1, 'Output directory is required'),

  resolution: z
    .number()
    .int('Resolution must be an integer')
    .min(256, 'Resolution must be at least 256')
    .max(4096, 'Resolution must not exceed 4096')
    .refine((val) => val % 64 === 0, {
      message: 'Resolution must be divisible by 64',
    }),

  num_repeats: z.number().int().min(1, 'Number of repeats must be at least 1').max(1000),

  max_train_epochs: z.number().int().min(0, 'Epochs must be non-negative').max(10000),

  max_train_steps: z.number().int().min(0, 'Steps must be non-negative'),

  train_batch_size: z.number().int().min(1, 'Batch size must be at least 1').max(128),

  seed: z.number().int().min(0, 'Seed must be non-negative'),

  // Data augmentation (booleans)
  flip_aug: z.boolean(),
  random_crop: z.boolean(),
  color_aug: z.boolean(),
  shuffle_caption: z.boolean(),

  // ========== LEARNING RATES ==========
  unet_lr: z
    .number()
    .positive('UNet learning rate must be positive')
    .max(1, 'UNet learning rate seems too high (max 1.0)'),

  text_encoder_lr: z
    .number()
    .nonnegative('Text encoder learning rate must be non-negative')
    .max(1, 'Text encoder learning rate seems too high (max 1.0)'),

  lr_scheduler: LRSchedulerSchema,

  lr_scheduler_number: z.number().int().min(0).max(100),

  lr_warmup_ratio: z.number().min(0, 'Warmup ratio must be non-negative').max(1, 'Warmup ratio must not exceed 1.0'),

  lr_warmup_steps: z.number().int().min(0, 'Warmup steps must be non-negative'),

  lr_power: z.number().min(0).max(10),

  // ========== LORA STRUCTURE ==========
  lora_type: LoRATypeSchema,

  network_module: z.string().min(1, 'Network module is required'),

  network_dim: z.number().int().min(1, 'Network dimension must be at least 1').max(1024),

  network_alpha: z.number().int().min(1, 'Network alpha must be at least 1').max(1024),

  conv_dim: z.number().int().min(1).max(1024),

  conv_alpha: z.number().int().min(1).max(1024),

  network_dropout: z.number().min(0, 'Dropout must be non-negative').max(1, 'Dropout must not exceed 1.0'),

  dim_from_weights: z.boolean(),

  factor: z.number().int().min(-1),

  train_norm: z.boolean(),

  rank_dropout: z.number().min(0).max(1),

  module_dropout: z.number().min(0).max(1),

  // Block-wise parameters (optional strings)
  down_lr_weight: z.string().optional(),
  mid_lr_weight: z.string().optional(),
  up_lr_weight: z.string().optional(),
  block_lr_zero_threshold: z.string().optional(),
  block_dims: z.string().optional(),
  block_alphas: z.string().optional(),
  conv_block_dims: z.string().optional(),
  conv_block_alphas: z.string().optional(),

  // ========== OPTIMIZER ==========
  optimizer_type: OptimizerSchema,

  weight_decay: z.number().min(0, 'Weight decay must be non-negative').max(1),

  gradient_accumulation_steps: z.number().int().min(1, 'Gradient accumulation must be at least 1').max(128),

  max_grad_norm: z.number().positive('Max gradient norm must be positive').max(10),

  optimizer_args: z.string().optional(),

  // ========== CAPTION & TOKEN CONTROL ==========
  keep_tokens: z.number().int().min(0),

  clip_skip: z.number().int().min(0).max(12),

  max_token_length: z.number().int().min(75).max(512),

  caption_dropout_rate: z.number().min(0).max(1),

  caption_tag_dropout_rate: z.number().min(0).max(1),

  caption_dropout_every_n_epochs: z.number().int().min(0),

  keep_tokens_separator: z.string(),

  secondary_separator: z.string(),

  enable_wildcard: z.boolean(),

  weighted_captions: z.boolean(),

  // ========== BUCKETING ==========
  enable_bucket: z.boolean(),

  sdxl_bucket_optimization: z.boolean().optional(),

  min_bucket_reso: z.number().int().min(64).max(2048),

  max_bucket_reso: z.number().int().min(256).max(4096),

  bucket_reso_steps: z.number().int().min(8).max(128).optional(),

  bucket_no_upscale: z.boolean(),

  // ========== ADVANCED TRAINING ==========
  min_snr_gamma_enabled: z.boolean().optional(),

  min_snr_gamma: z.number().min(0).max(20),

  ip_noise_gamma_enabled: z.boolean().optional(),

  ip_noise_gamma: z.number().min(0).max(1),

  multinoise: z.boolean().optional(),

  multires_noise_discount: z.number().min(0).max(1),

  noise_offset: z.number().min(0).max(1),

  adaptive_noise_scale: z.number().min(0).max(1),

  zero_terminal_snr: z.boolean().optional(),

  // ========== MEMORY & PERFORMANCE ==========
  gradient_checkpointing: z.boolean().optional(),

  mixed_precision: MixedPrecisionSchema,

  full_fp16: z.boolean(),

  full_bf16: z.boolean().optional(),

  fp8_base: z.boolean(),

  vae_batch_size: z.number().int().min(0),

  no_half_vae: z.boolean(),

  cache_latents: z.boolean(),

  cache_latents_to_disk: z.boolean(),

  cache_text_encoder_outputs: z.boolean(),

  cache_text_encoder_outputs_to_disk: z.boolean().optional(),

  cross_attention: z.string().optional(),

  persistent_data_loader_workers: z.number().int().min(0),

  no_token_padding: z.boolean().optional(),

  // ========== SAVING & CHECKPOINTS ==========
  save_every_n_epochs: z.number().int().min(0),

  save_every_n_steps: z.number().int().min(0),

  save_last_n_epochs: z.number().int().min(0),

  save_last_n_epochs_state: z.number().int().min(0),

  save_state: z.boolean(),

  save_state_on_train_end: z.boolean().optional(),

  save_last_n_steps_state: z.number().int().min(0).optional(),

  save_model_as: SaveFormatSchema,

  save_precision: z.string(),

  output_name: z.string().optional(),

  no_metadata: z.boolean().optional(),

  resume_from_state: z.string().optional(),

  // ========== SAMPLE GENERATION ==========
  sample_every_n_epochs: z.number().int().min(0),

  sample_every_n_steps: z.number().int().min(0),

  sample_prompts: z.string().optional(),

  sample_sampler: z.string(),

  // ========== LOGGING ==========
  logging_dir: z.string().optional(),

  log_with: z.string().optional(),

  log_prefix: z.string().optional(),

  log_tracker_name: z.string().optional(),

  log_tracker_config: z.string().optional(),

  wandb_run_name: z.string().optional(),

  // ========== SD 2.x & ADVANCED ==========
  v2: z.boolean().optional(),

  v_parameterization: z.boolean().optional(),

  network_train_unet_only: z.boolean().optional(),

  prior_loss_weight: z.number().min(0).max(1).optional(),

  // Additional advanced parameters
  scale_v_pred_loss_like_noise_pred: z.boolean().optional(),
  v_pred_like_loss: z.number().min(0).optional(),
  debiased_estimation_loss: z.boolean().optional(),
  lowram: z.boolean().optional(),
  max_data_loader_n_workers: z.number().int().min(0).optional(),
  min_timestep: z.number().int().min(0).optional(),
  max_timestep: z.number().int().min(0).optional(),
  multires_noise_iterations: z.number().int().min(0).optional(),
  ip_noise_gamma_random_strength: z.boolean().optional(),

  // ========== FLUX-SPECIFIC PARAMETERS ==========
  ae_path: z.string().optional(),

  t5xxl_max_token_length: z.number().int().min(128).max(1024).optional(),

  apply_t5_attn_mask: z.boolean().optional(),

  guidance_scale: z.number().min(0).max(30).optional(),

  timestep_sampling: z.string().optional(),

  sigmoid_scale: z.number().min(0).optional(),

  model_prediction_type: z.string().optional(),

  blocks_to_swap: z.number().int().min(0).optional(),

  // ========== LUMINA-SPECIFIC PARAMETERS ==========
  gemma2: z.string().optional(),

  gemma2_max_token_length: z.number().int().min(128).max(1024).optional(),
});

/**
 * Partial schema for updating configuration
 * Allows updating individual fields without requiring all fields
 */
export const PartialTrainingConfigSchema = TrainingConfigSchema.partial();

/**
 * Type inference from schema
 */
export type ValidatedTrainingConfig = z.infer<typeof TrainingConfigSchema>;

/**
 * Helper function to validate training configuration
 * Returns validation result with detailed error messages
 */
export function validateTrainingConfig(config: unknown) {
  return TrainingConfigSchema.safeParse(config);
}

/**
 * Helper function to validate partial configuration updates
 */
export function validatePartialConfig(config: unknown) {
  return PartialTrainingConfigSchema.safeParse(config);
}

/**
 * Extract validation errors into user-friendly format
 */
export function formatValidationErrors(errors: z.ZodError) {
  return errors.errors.map((error) => ({
    field: error.path.join('.'),
    message: error.message,
    code: error.code,
  }));
}
