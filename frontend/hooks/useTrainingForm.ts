/**
 * Training form hook - RHF + localStorage (no Zustand)
 *
 * Single source of truth: React Hook Form
 * Persistence: localStorage directly
 * No middle-man store to create race conditions.
 */

import { useForm, UseFormReturn } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useCallback, useEffect, useRef, useState } from 'react';
import { TrainingConfigSchema } from '@/lib/validation';
import type { TrainingConfig } from '@/lib/api';

const STORAGE_KEY = 'training-config';

/**
 * Default training configuration values
 */
const defaultConfig: TrainingConfig = {
  project_name: 'my_lora',
  model_type: 'SDXL',
  pretrained_model_name_or_path: '',
  guidance_scale: 3.5,
  timestep_sampling: 'sigma',
  sigmoid_scale: 1.0,
  model_prediction_type: 'raw',
  apply_t5_attn_mask: false,
  vae_path: '',
  train_data_dir: '',
  output_dir: '',
  resolution: 1024,
  num_repeats: 10,
  max_train_epochs: 10,
  max_train_steps: 0,
  train_batch_size: 1,
  seed: 42,
  flip_aug: false,
  random_crop: false,
  color_aug: false,
  shuffle_caption: false,
  unet_lr: 0.0001,
  text_encoder_lr: 0.00001,
  lr_scheduler: 'cosine_with_restarts',
  lr_scheduler_number: 3,
  lr_warmup_ratio: 0.1,
  lr_warmup_steps: 0,
  lr_power: 1.0,
  lora_type: 'LoRA',
  network_module: 'networks.lora',
  network_dim: 32,
  network_alpha: 32,
  conv_dim: 0,
  conv_alpha: 0,
  network_dropout: 0.0,
  dim_from_weights: false,
  factor: -1,
  train_norm: false,
  rank_dropout: 0.0,
  module_dropout: 0.0,
  optimizer_type: 'AdamW8bit',
  weight_decay: 0.01,
  gradient_accumulation_steps: 1,
  max_grad_norm: 1.0,
  keep_tokens: 0,
  clip_skip: 2,
  max_token_length: 225,
  caption_dropout_rate: 0.0,
  caption_tag_dropout_rate: 0.0,
  caption_dropout_every_n_epochs: 0,
  keep_tokens_separator: '|||',
  secondary_separator: '',
  caption_extension: '.txt',
  enable_wildcard: false,
  weighted_captions: false,
  enable_bucket: true,
  min_bucket_reso: 256,
  max_bucket_reso: 2048,
  bucket_no_upscale: false,
  bucket_reso_steps: 64,
  mixed_precision: 'fp16',
  save_precision: 'fp16',
  full_fp16: false,
  full_bf16: false,
  fp8_base: false,
  cache_latents: true,
  cache_latents_to_disk: false,
  cache_text_encoder_outputs: false,
  cache_text_encoder_outputs_to_disk: false,
  no_half_vae: false,
  save_model_as: 'safetensors',
  save_every_n_epochs: 1,
  save_every_n_steps: 0,
  save_last_n_epochs: 0,
  save_last_n_epochs_state: 0,
  save_state: false,
  save_state_on_train_end: false,
  resume_from_state: '',
  logging_dir: '',
  log_with: '',
  log_prefix: '',
  sample_every_n_epochs: 0,
  sample_every_n_steps: 0,
  sample_prompts: '',
  sample_sampler: 'euler_a',
  min_snr_gamma: 5.0,
  scale_v_pred_loss_like_noise_pred: false,
  v_pred_like_loss: 0,
  debiased_estimation_loss: false,
  noise_offset: 0,
  adaptive_noise_scale: 0,
  ip_noise_gamma: 0,
  ip_noise_gamma_random_strength: false,
  multires_noise_iterations: 0,
  multires_noise_discount: 0.0,
  gradient_checkpointing: true,
  cross_attention: 'sdpa',
  no_token_padding: false,
  lowram: false,
  max_data_loader_n_workers: 8,
  persistent_data_loader_workers: 0,
  vae_batch_size: 1,
  min_timestep: 0,
  max_timestep: 1000,
  ae_path: '',
  v_parameterization: false,
  // Optional fields with defaults to prevent uncontrolled→controlled warnings
  training_mode: 'lora' as const,
  v2: false,
  min_snr_gamma_enabled: false,
  ip_noise_gamma_enabled: false,
  multinoise: false,
  zero_terminal_snr: false,
  no_metadata: false,
  save_last_n_steps_state: 0,
  output_name: 'lora',
  continue_from_lora: '',
  wandb_key: '',
  clip_l_path: '',
  clip_g_path: '',
  t5xxl_path: '',
  blocks_to_swap: 0,
  gemma2: '',
  log_tracker_name: '',
  log_tracker_config: '',
  wandb_run_name: '',
};

/** Coerce string values back to numbers after JSON parse from localStorage */
function coerceNumericFields(config: Record<string, any>): void {
  const floatFields = [
    'unet_lr', 'text_encoder_lr', 'lr_warmup_ratio', 'lr_power',
    'network_dropout', 'rank_dropout', 'module_dropout', 'weight_decay',
    'max_grad_norm', 'caption_dropout_rate', 'caption_tag_dropout_rate',
    'noise_offset', 'adaptive_noise_scale', 'ip_noise_gamma',
    'multires_noise_discount', 'guidance_scale', 'sigmoid_scale',
  ];
  for (const field of floatFields) {
    if (field in config && typeof config[field] === 'string') {
      config[field] = Number(config[field]) || 0;
    }
  }

  const intFields = [
    'resolution', 'num_repeats', 'max_train_epochs', 'max_train_steps',
    'train_batch_size', 'seed', 'network_dim', 'network_alpha',
    'conv_dim', 'conv_alpha', 'gradient_accumulation_steps',
    'keep_tokens', 'clip_skip', 'max_token_length', 'lr_scheduler_number',
    'lr_warmup_steps', 'save_every_n_epochs', 'save_every_n_steps',
    'min_bucket_reso', 'max_bucket_reso', 'bucket_reso_steps',
    'vae_batch_size', 'factor',
  ];
  for (const field of intFields) {
    if (field in config && typeof config[field] === 'string') {
      config[field] = parseInt(config[field], 10) || 0;
    }
  }
}

/** Read config from localStorage, handling both old Zustand format and new direct format */
function readStoredConfig(): TrainingConfig | null {
  if (typeof window === 'undefined') return null;

  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;

    const parsed = JSON.parse(raw);

    // Old Zustand persist format: { state: { config: {...} } }
    // New direct format: just the config object
    const config = parsed?.state?.config || parsed;

    if (!config || typeof config !== 'object') return null;

    // Merge with defaults to fill any missing fields
    const merged = { ...defaultConfig, ...config };
    coerceNumericFields(merged);
    return merged;
  } catch (e) {
    console.error('Failed to read training config from localStorage:', e);
    return null;
  }
}

/** Write config to localStorage */
function writeStoredConfig(config: TrainingConfig): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
  } catch (e) {
    console.error('Failed to save training config to localStorage:', e);
  }
}

/**
 * Custom hook for managing the training configuration form.
 *
 * Wraps React Hook Form with a Zod resolver, hydrates initial values from
 * localStorage on mount, and optionally debounces auto-saves on every change.
 * Validation mode is `"onChange"` when `validateOnChange` is `true`, and
 * `"onSubmit"` otherwise (the default, to prevent mid-typing resets).
 *
 * `@param` options                  - Hook configuration options.
 * `@param` options.autoSave         - Auto-save form state to localStorage on change. Defaults to `true`.
 * `@param` options.autoSaveDelay    - Debounce delay in ms for auto-save. Defaults to `500`.
 * `@param` options.validateOnChange - Run Zod validation on every keystroke when `true`. Defaults to `false`.
 *
 * `@returns`
 * - `form`                - The RHF `UseFormReturn<TrainingConfig>` instance.
 * - `config`              - Snapshot of current form values (for read-only consumers like `PresetManager`).
 * - `isDirty`             - `true` when the form has changes not yet written to localStorage.
 * - `isValid`             - `true` when all fields pass Zod validation.
 * - `isHydrated`          - `true` once localStorage hydration has completed.
 * - `syncToStore`         - Immediately flushes current values to localStorage (no debounce).
 * - `loadPreset`          - Merges a partial config preset over current values and resets the form.
 * - `onSubmit`            - Returns an RHF submit handler that syncs to localStorage before invoking the callback.
 * - `getValidationErrors` - Returns a flat `{ field, message }[]` array of current validation errors.
 * - `resetForm`           - Resets the form to `defaultConfig` and clears localStorage.
 */
export function useTrainingForm(options: {
  autoSave?: boolean;
  autoSaveDelay?: number;
  validateOnChange?: boolean;
} = {}) {
  const { autoSave = true, autoSaveDelay = 500, validateOnChange = false } = options;

  const [isHydrated, setIsHydrated] = useState(false);
  const isHydratedRef = useRef(false);

  // Initialize form with hardcoded defaults (localStorage hydration happens in useEffect)
  const form = useForm<TrainingConfig>({
  resolver: zodResolver(TrainingConfigSchema) as any,
  defaultValues: defaultConfig,
  shouldUnregister: false,
	mode: validateOnChange ? "onChange" : "onSubmit",  // 👈 Dynamic mode
  });

  const formRef = useRef(form);
  formRef.current = form;

  // 1. Hydrate from localStorage ONCE on client mount
  useEffect(() => {
    if (isHydratedRef.current) return;

    const stored = readStoredConfig();
    if (stored) {
      formRef.current.reset(stored, { keepDefaultValues: false });
      console.log('Hydrated form from localStorage:', stored.project_name);
    }

    isHydratedRef.current = true;
    setIsHydrated(true);
  }, []);

  // 2. Auto-save: watch form changes → write to localStorage (debounced)
  useEffect(() => {
    if (!autoSave || !isHydrated) return;

    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const subscription = form.watch(() => {
      if (timeoutId) clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        const values = form.getValues();
        writeStoredConfig(values);
      }, autoSaveDelay);
    });

    return () => {
      subscription.unsubscribe();
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [autoSave, autoSaveDelay, form, isHydrated]);

  // Force-save to localStorage (immediate, no debounce)
  const syncToStore = useCallback(() => {
    const values = form.getValues();
    writeStoredConfig(values);
  }, [form]);

  const loadPreset = useCallback((preset: Partial<TrainingConfig>) => {
    if (!preset || typeof preset !== 'object') {
      console.error('Invalid preset:', preset);
      return;
    }

    // Merge preset over current form values (preserves paths, project name, etc.)
    const currentValues = form.getValues();
    const fullConfig = { ...currentValues, ...preset } as TrainingConfig;

    // Update form and save to localStorage
    form.reset(fullConfig, { keepDefaultValues: false, keepDirty: false, keepValues: false });
    writeStoredConfig(fullConfig);

    console.log('Preset loaded:', fullConfig.optimizer_type);
  }, [form]);

  const getValidationErrors = useCallback(() => {
    const errors = form.formState.errors;
    return Object.entries(errors).map(([field, error]) => ({
      field,
      message: (error as any)?.message || 'Invalid value',
    }));
  }, [form.formState.errors]);

  const onSubmit = useCallback(
    (callback: (config: TrainingConfig) => void) => {
      return form.handleSubmit((data) => {
        // Force sync to localStorage before training starts
        writeStoredConfig(data);
        callback(data);
      });
    },
    [form]
  );

  return {
    form: form as unknown as UseFormReturn<TrainingConfig>,
    // config: current form values for components that need them (e.g. PresetManager)
    config: form.getValues() as TrainingConfig,
    isDirty: form.formState.isDirty,
    isValid: form.formState.isValid,
    isHydrated,
    syncToStore,
    loadPreset,
    onSubmit,
    getValidationErrors,
    resetForm: () => {
      form.reset(defaultConfig, { keepDefaultValues: false });
      writeStoredConfig(defaultConfig);
    },
  };
}

/**
 * Helper hook for form field registration with error display
 */
export function useFormField(
  name: keyof TrainingConfig,
  form: UseFormReturn<TrainingConfig>
) {
  const register = form.register(name);
  const error = (form.formState.errors as any)[name]?.message;

  return {
    register,
    error,
    hasError: !!error,
  };
}

/**
 * Configuration presets for common training scenarios
 */
export const trainingPresets: Record<string, {
  name: string;
  description: string;
  config: Partial<TrainingConfig>
}> = {
  sdxl_character: {
    name: 'SDXL Character',
    description: 'Optimized for training character LoRAs on SDXL',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 32,
      network_alpha: 32,
      unet_lr: 0.0001,
      text_encoder_lr: 0.00005,
      lr_scheduler: 'cosine_with_restarts',
      lr_scheduler_number: 3,
      optimizer_type: 'AdamW8bit' as any,
      max_train_epochs: 10,
      train_batch_size: 1,
      clip_skip: 2,
    },
  },
  sdxl_style: {
    name: 'SDXL Style',
    description: 'For training art style LoRAs on SDXL',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 64,
      network_alpha: 64,
      unet_lr: 0.0002,
      text_encoder_lr: 0.0001,
      lr_scheduler: 'cosine',
      optimizer_type: 'Prodigy' as any,
      max_train_epochs: 15,
      train_batch_size: 2,
      clip_skip: 2,
    },
  },
  sd15_character: {
    name: 'SD1.5 Character',
    description: 'Classic SD1.5 character training',
    config: {
      model_type: 'SD1.5',
      resolution: 512,
      network_dim: 32,
      network_alpha: 32,
      unet_lr: 0.0001,
      text_encoder_lr: 0.00005,
      lr_scheduler: 'cosine_with_restarts',
      lr_scheduler_number: 3,
      optimizer_type: 'AdamW8bit' as any,
      max_train_epochs: 10,
      train_batch_size: 4,
      clip_skip: 1,
    },
  },
  flux_experimental: {
    name: 'Flux (Experimental)',
    description: 'Experimental Flux.1 training',
    config: {
      model_type: 'Flux',
      resolution: 1024,
      network_dim: 16,
      network_alpha: 16,
      unet_lr: 0.0001,
      text_encoder_lr: 0,
      lr_scheduler: 'constant',
      optimizer_type: 'AdamW8bit' as any,
      max_train_epochs: 5,
      train_batch_size: 1,
      guidance_scale: 3.5,
      timestep_sampling: 'flux_shift',
    },
  },
  kappaneuro_mucha_style: {
    name: "KappaNeuro's Mucha Style (SDXL)",
    description: 'KappaNeuro Alphonse Mucha style LoRA. Extremely fast — 3 epochs, 16 min, batch 8. dim 32/alpha 8 (4:1 conservative), frozen TE, cosine, low noise_offset=0.03.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 32,
      network_alpha: 8,
      unet_lr: 0.0004,
      text_encoder_lr: 0,
      lr_scheduler: 'cosine',
      optimizer_type: 'AdamW8bit' as any,
      optimizer_args: 'weight_decay=0.1 betas=(0.9,0.99)',
      max_train_epochs: 3,
      save_every_n_epochs: 1,
      train_batch_size: 8,
      mixed_precision: 'bf16',
      noise_offset: 0.03,
    },
  },
  add_details_xl: {
    name: 'Add Details XL (Utility LoRA)',
    description: 'Detail-enhancement utility LoRA approach. 29 images × 30 repeats, 1 epoch only, triple dropout (rank+module+network=0.1) to prevent memorization. Uses HuggingFace model ID directly.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 64,
      network_alpha: 32,
      unet_lr: 0.0001,
      text_encoder_lr: 0.0001,
      lr_scheduler: 'cosine_with_restarts',
      optimizer_type: 'AdamW8bit' as any,
      max_train_epochs: 1,
      save_every_n_epochs: 1,
      train_batch_size: 1,
      mixed_precision: 'bf16',
      noise_offset: 0.1,
      min_snr_gamma: 5,
      min_snr_gamma_enabled: true,
      network_dropout: 0.1,
      rank_dropout: 0.1,
      module_dropout: 0.1,
    },
  },
  faetastic_sdxl_prodigy: {
    name: 'FaeTastic Style (SDXL Prodigy)',
    description: 'FaeTastic SDXL style LoRA. Prodigy with cosine (unusual — decay stacks on adaptation), weight_decay=0.5 + bias_correction=False, unified LR=1. noise_offset=0.0357. dim 64/64, 100 images × 5 repeats.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 64,
      network_alpha: 64,
      unet_lr: 1,
      text_encoder_lr: 1,
      lr_scheduler: 'cosine',
      optimizer_type: 'Prodigy',
      optimizer_args: 'decouple=True weight_decay=0.5 betas=(0.9,0.99) use_bias_correction=False',
      max_train_epochs: 20,
      save_every_n_epochs: 2,
      train_batch_size: 1,
      mixed_precision: 'bf16',
      noise_offset: 0.0357,
    },
  },
  lah_cutedoodle_sd15: {
    name: "Lah's Cutedoodle (SD1.5)",
    description: "Lah's cute doodle style on SD1.5/AnythingV5. dim 64/32, 768px, constant scheduler, adaptive_noise_scale=0.00357 (1/10 of noise_offset). Stopped at epoch 8/10.",
    config: {
      model_type: 'SD1.5',
      resolution: 768,
      network_dim: 64,
      network_alpha: 32,
      unet_lr: 0.0001,
      text_encoder_lr: 0.00005,
      lr_scheduler: 'constant',
      optimizer_type: 'AdamW8bit' as any,
      max_train_epochs: 10,
      save_every_n_epochs: 1,
      train_batch_size: 4,
      mixed_precision: 'fp16',
      noise_offset: 0.05,
      adaptive_noise_scale: 0.00357,
    },
  },
  lah_cutedoodle_xl: {
    name: "Lah's Cutedoodle (XL)",
    description: "Lah's cute doodle style on SDXL. TE LR (5e-4) HIGHER than UNet (1e-4) — inverted convention. Plain AdamW (not 8bit), constant, noise_offset=0.0357, adaptive_noise_scale=0.00357. Stopped epoch 12/15.",
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 32,
      network_alpha: 32,
      unet_lr: 0.0001,
      text_encoder_lr: 0.0005,
      lr_scheduler: 'constant',
      optimizer_type: 'AdamW',
      max_train_epochs: 15,
      save_every_n_epochs: 1,
      train_batch_size: 8,
      mixed_precision: 'bf16',
      noise_offset: 0.0357,
      adaptive_noise_scale: 0.00357,
    },
  },
  folk_horror_style_adafactor: {
    name: 'Folk Horror Style (SDXL Adafactor)',
    description: 'SDXL style LoRA with Adafactor fixed-LR mode. dim 32/16, active TE at 5e-5, noise_offset=0.1, SNR=5, batch 4. Essentially the Adafactor equivalent of Style Baking.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 32,
      network_alpha: 16,
      unet_lr: 0.0005,
      text_encoder_lr: 0.00005,
      lr_scheduler: 'cosine_with_restarts',
      optimizer_type: 'AdaFactor',
      optimizer_args: 'scale_parameter=False relative_step=False warmup_init=False',
      max_train_epochs: 10,
      save_every_n_epochs: 1,
      train_batch_size: 4,
      mixed_precision: 'bf16',
      noise_offset: 0.1,
      min_snr_gamma: 5,
      min_snr_gamma_enabled: true,
    },
  },
  flat_color_anime_xl: {
    name: 'Flat Color Anime Style (Base SDXL)',
    description: 'Base SDXL flat color anime style. Extreme dim 128/alpha 1 ratio (1:128 scaling), Adafactor fixed-LR mode, active TE at 5e-5, stopped at epoch 23/25. Long run (~8h).',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 128,
      network_alpha: 1,
      unet_lr: 0.0001,
      text_encoder_lr: 0.00005,
      lr_scheduler: 'cosine_with_restarts',
      optimizer_type: 'AdaFactor',
      optimizer_args: 'scale_parameter=False relative_step=False warmup_init=False',
      max_train_epochs: 25,
      save_every_n_epochs: 1,
      train_batch_size: 5,
      mixed_precision: 'bf16',
    },
  },
  anything_sd15_historical_2023: {
    name: 'Anything v4.5 (SD1.5 — Historical 2023)',
    description: 'June 2023 Anything v4.5 LoRA. HISTORICAL — alpha 128 / dim 32 (4:1 inverse, very aggressive scaling). Only ran 1/20 epochs (2 min) — likely a test or crash. Kept as historical curiosity.',
    config: {
      model_type: 'SD1.5',
      resolution: 512,
      network_dim: 32,
      network_alpha: 128,
      unet_lr: 0.0001,
      text_encoder_lr: 0.00001,
      lr_scheduler: 'cosine_with_restarts',
      optimizer_type: 'AdamW8bit' as any,
      max_train_epochs: 20,
      save_every_n_epochs: 1,
      train_batch_size: 2,
      mixed_precision: 'fp16',
    },
  },
  pastel_style_sd21_historical: {
    name: 'Pastel Style (SD 2.1 — Historical)',
    description: 'Feb 2023 Platdiffusion/SD 2.1 pastel style LoRA. HISTORICAL — predates standardized metadata. dim 96/alpha 12 (1:8 ratio), UNet 2× TE LR, 512px. WARNING: SD 2.1 not in model type enum — needs v2=true and OpenCLIP ViT-H. Use SD1.5 slot as workaround.',
    config: {
      model_type: 'SD1.5',
      resolution: 512,
      network_dim: 96,
      network_alpha: 12,
      unet_lr: 0.0004,
      text_encoder_lr: 0.0002,
      lr_scheduler: 'cosine',
      optimizer_type: 'AdamW',
      max_train_epochs: 6,
      save_every_n_epochs: 1,
      train_batch_size: 2,
      mixed_precision: 'fp16',
    },
  },
  tensorart_dora_illustrious: {
    name: "TensorArt DoRA (Illustrious)",
    description: "Illustrious DoRA via lycoris.kohya (algo=lora + dora_wd=True). Larger dim 64/32, frozen TE, constant_with_warmup, noise_offset=0.03. Note: original used a server-side lycoris preset.toml and factor=8 — omitted here.",
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      lora_type: 'DoRA',
      network_module: 'lycoris.kohya',
      network_dim: 64,
      network_alpha: 32,
      unet_lr: 0.0001,
      text_encoder_lr: 0,
      lr_scheduler: 'constant_with_warmup',
      optimizer_type: 'AdamW8bit' as any,
      max_train_epochs: 10,
      save_every_n_epochs: 1,
      train_batch_size: 2,
      mixed_precision: 'bf16',
      noise_offset: 0.03,
    },
  },
  came_character_noob_batch6: {
    name: 'CAME Character (NoobAI, batch 6)',
    description: 'CAME character LoRA for NoobAI epsilon. Higher batch (6), SNR=8, UNet LR (6e-5) lower than base LR (1e-4). Stopped at epoch 10 of 15 — pick best checkpoint.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 8,
      network_alpha: 4,
      unet_lr: 0.00006,
      text_encoder_lr: 0.000001,
      lr_scheduler: 'cosine',
      optimizer_type: 'CAME',
      optimizer_args: 'weight_decay=0.04',
      max_train_epochs: 15,
      save_every_n_epochs: 1,
      train_batch_size: 6,
      mixed_precision: 'bf16',
      min_snr_gamma: 8,
      min_snr_gamma_enabled: true,
    },
  },
  compass_creature_noob: {
    name: 'Compass Attn-MLP (NoobAI)',
    description: 'NoobAI LyCORIS locon restricted to attn+MLP layers only (conv_dim=0, preset=attn-mlp), high LR, low SNR=1.5, effective batch 32. NOTE: Original used Compass optimizer (library.stochastic_optim) — mapped to AdamW until sourced.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      lora_type: 'LoCon',
      network_module: 'lycoris.kohya',
      network_dim: 16,
      network_alpha: 16,
      conv_dim: 0,
      conv_alpha: 0,
      unet_lr: 0.0006,
      text_encoder_lr: 0.0001,
      lr_scheduler: 'cosine',
      optimizer_type: 'AdamW',
      optimizer_args: 'weight_decay=0 betas=(0.9,0.999)',
      max_train_epochs: 6,
      save_every_n_epochs: 1,
      train_batch_size: 16,
      gradient_accumulation_steps: 2,
      mixed_precision: 'bf16',
      min_snr_gamma: 1.5,
      min_snr_gamma_enabled: true,
    },
  },
  came_character_pony: {
    name: 'CAME Character (Pony)',
    description: 'CAME-based character LoRA for Pony. Small dim 8/4, near-frozen TE (1e-6), cosine, SNR=5, batch 2.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 8,
      network_alpha: 4,
      unet_lr: 0.0001,
      text_encoder_lr: 0.000001,
      lr_scheduler: 'cosine',
      optimizer_type: 'CAME',
      optimizer_args: 'weight_decay=0.04',
      max_train_epochs: 10,
      save_every_n_epochs: 1,
      train_batch_size: 2,
      mixed_precision: 'bf16',
      min_snr_gamma: 5,
      min_snr_gamma_enabled: true,
    },
  },
  came_character_illustrious: {
    name: 'CAME Character (Illustrious)',
    description: 'CAME-based character LoRA for Illustrious. Same as Pony variant but batch 1, no SNR.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 8,
      network_alpha: 4,
      unet_lr: 0.0001,
      text_encoder_lr: 0.000001,
      lr_scheduler: 'cosine',
      optimizer_type: 'CAME',
      optimizer_args: 'weight_decay=0.04',
      max_train_epochs: 10,
      save_every_n_epochs: 1,
      train_batch_size: 1,
      mixed_precision: 'bf16',
    },
  },
  came_character_initium: {
    name: 'CAME Character (Initium/NoobAI finetune)',
    description: 'CAME character LoRA for Initium (NoobAI finetune). 10× lower LR (3e-5), more epochs, noise_offset=0.0357.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 8,
      network_alpha: 4,
      unet_lr: 0.00003,
      text_encoder_lr: 0.000007,
      lr_scheduler: 'cosine',
      optimizer_type: 'CAME',
      optimizer_args: 'weight_decay=0.04',
      max_train_epochs: 17,
      save_every_n_epochs: 1,
      train_batch_size: 1,
      mixed_precision: 'bf16',
      noise_offset: 0.0357,
    },
  },
  enigmata_character_noobai_vpred: {
    name: "Enigmata's Character (NoobAI v-pred)",
    description: 'Published character LoRA by Enigmata (Igawa Asagi). NoobAI v-prediction, slightly lower LRs than epsilon version (1.2e-4 vs 1.4e-4), same multi-subset outfit dataset.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 32,
      network_alpha: 16,
      unet_lr: 0.00012,
      text_encoder_lr: 0.00006,
      lr_scheduler: 'cosine_with_restarts',
      optimizer_type: 'AdamW8bit' as any,
      optimizer_args: 'weight_decay=0.1 betas=(0.9,0.99)',
      max_train_epochs: 13,
      save_every_n_epochs: 1,
      train_batch_size: 4,
      mixed_precision: 'bf16',
      v_parameterization: true,
      zero_terminal_snr: true,
      scale_v_pred_loss_like_noise_pred: true,
      min_snr_gamma: 5,
      min_snr_gamma_enabled: true,
    },
  },
  enigmata_character_noobai: {
    name: "Enigmata's Character (NoobAI)",
    description: 'Published character LoRA by Enigmata (Igawa Asagi). NoobAI epsilon, multi-subset outfit dataset, LR 1.4e-4 with betas=(0.9,0.99), SNR=5.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 32,
      network_alpha: 16,
      unet_lr: 0.00014,
      text_encoder_lr: 0.00007,
      lr_scheduler: 'cosine_with_restarts',
      optimizer_type: 'AdamW8bit' as any,
      optimizer_args: 'weight_decay=0.1 betas=(0.9,0.99)',
      max_train_epochs: 13,
      save_every_n_epochs: 1,
      train_batch_size: 4,
      mixed_precision: 'bf16',
      min_snr_gamma: 5,
      min_snr_gamma_enabled: true,
    },
  },
  enigmata_character_pony: {
    name: "Enigmata's Character (Pony)",
    description: 'Published character LoRA by Enigmata (Igawa Asagi). Pony base, smaller dim (16/8), lower LR 8e-5, no SNR, multi-subset outfit dataset.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 16,
      network_alpha: 8,
      unet_lr: 0.00008,
      text_encoder_lr: 0.00004,
      lr_scheduler: 'cosine_with_restarts',
      optimizer_type: 'AdamW8bit' as any,
      optimizer_args: 'weight_decay=0.1 betas=(0.9,0.99)',
      max_train_epochs: 11,
      save_every_n_epochs: 1,
      train_batch_size: 4,
      mixed_precision: 'bf16',
    },
  },
  enigmata_character_sd15: {
    name: "Enigmata's Character (SD1.5)",
    description: 'Published character LoRA by Enigmata (Igawa Asagi). SD1.5 base at 720px, large dim 64/32, higher LR 2e-4, balanced multi-subset repeats by outfit rarity.',
    config: {
      model_type: 'SD1.5',
      resolution: 720,
      network_dim: 64,
      network_alpha: 32,
      unet_lr: 0.0002,
      text_encoder_lr: 0.0001,
      lr_scheduler: 'cosine_with_restarts',
      optimizer_type: 'AdamW8bit' as any,
      max_train_epochs: 11,
      save_every_n_epochs: 1,
      train_batch_size: 2,
      mixed_precision: 'fp16',
      noise_offset: 0.05,
    },
  },
  ktiseos_nyx_illustrious_character: {
    name: "Ktiseos Nyx's Character (Illustrious)",
    description: "Published Illustrious character LoRA recipe by ktiseos_nyx. AdamW8Bit with weight_decay=0.1, small active TE (5e-5), batch 3, multinoise, SNR=5. Solid baseline for character training.",
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 32,
      network_alpha: 16,
      unet_lr: 0.0005,
      text_encoder_lr: 0.00005,
      lr_scheduler: 'cosine_with_restarts',
      optimizer_type: 'AdamW8bit' as any,
      optimizer_args: 'weight_decay=0.1',
      max_train_epochs: 10,
      save_every_n_epochs: 1,
      train_batch_size: 3,
      mixed_precision: 'bf16',
      noise_offset: 0.1,
      multires_noise_iterations: 6,
      multires_noise_discount: 0.3,
      min_snr_gamma: 5,
      min_snr_gamma_enabled: true,
    },
  },
  flux_hard_bake_style: {
    name: 'Experimental Hard Bake Style (Flux)',
    description: "Flux.1 D style LoRA using lora_flux. Same dim:2/alpha:16 ratio as Chroma preset — frozen TE, AdamW8Bit. The parallel experiment to the Chroma version.",
    config: {
      model_type: 'Flux',
      resolution: 1024,
      network_module: 'networks.lora_flux',
      network_dim: 2,
      network_alpha: 16,
      unet_lr: 0.0005,
      text_encoder_lr: 0,
      lr_scheduler: 'cosine_with_restarts',
      optimizer_type: 'AdamW8bit' as any,
      optimizer_args: 'weight_decay=0.01 eps=1e-08 betas=(0.9,0.999)',
      max_train_epochs: 10,
      save_every_n_epochs: 1,
      train_batch_size: 2,
      mixed_precision: 'bf16',
      noise_offset: 0.1,
      multires_noise_iterations: 6,
      multires_noise_discount: 0.3,
      min_snr_gamma: 5,
      min_snr_gamma_enabled: true,
    },
  },
  bolero537_yukata_noob: {
    name: "bolero537's Yukata LoCon (NoobAI)",
    description: 'Published yukata clothing concept LoRA by bolero537. NoobAI epsilon, LoCon with extreme low alpha (16/1), Lion8bit, tiny dataset with heavy repeats (31 × 10).',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      lora_type: 'LoCon',
      network_module: 'lycoris.kohya',
      network_dim: 16,
      network_alpha: 1,
      conv_dim: 16,
      conv_alpha: 1,
      unet_lr: 0.0001,
      text_encoder_lr: 0,
      lr_scheduler: 'cosine_with_restarts',
      optimizer_type: 'Lion8bit',
      max_train_epochs: 13,
      save_every_n_epochs: 1,
      train_batch_size: 2,
      mixed_precision: 'bf16',
      min_snr_gamma: 6,
      min_snr_gamma_enabled: true,
    },
  },
  bolero537_yukata_pony: {
    name: "bolero537's Yukata (Pony)",
    description: 'Published yukata concept LoRA by bolero537. Pony base, Adafactor in adaptive mode (relative_step=True, self-scaling LR), extreme low alpha (8/1), same 31×10 dataset.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 8,
      network_alpha: 1,
      unet_lr: 0,
      text_encoder_lr: 0,
      lr_scheduler: 'constant',
      optimizer_type: 'AdaFactor',
      optimizer_args: 'relative_step=True warmup_init=False',
      max_train_epochs: 20,
      save_every_n_epochs: 1,
      train_batch_size: 2,
      mixed_precision: 'bf16',
      min_snr_gamma: 6,
      min_snr_gamma_enabled: true,
    },
  },
  bolero537_yukata_xl: {
    name: "bolero537's Yukata (Base XL)",
    description: 'Published yukata concept LoRA by bolero537. Custom SDXL mix base, AdamW8Bit cosine, unified LR 2e-5, standard 32/16 dim/alpha, same 31×10 dataset.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 32,
      network_alpha: 16,
      unet_lr: 0.00002,
      text_encoder_lr: 0,
      lr_scheduler: 'cosine',
      optimizer_type: 'AdamW8bit' as any,
      max_train_epochs: 10,
      save_every_n_epochs: 1,
      train_batch_size: 2,
      mixed_precision: 'bf16',
    },
  },
  motimalu_painterly_illustrious: {
    name: "motimalu's Painterly Styles (Illustrious)",
    description: 'Published Illustrious painterly style LoRA by motimalu. Prodigy with aggressive weight_decay=0.5 and bias_correction=False, large 1278-image dataset, 50 full epochs.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 32,
      network_alpha: 32,
      unet_lr: 1,
      text_encoder_lr: 1,
      lr_scheduler: 'constant',
      optimizer_type: 'Prodigy',
      optimizer_args: 'decouple=True weight_decay=0.5 betas=(0.9,0.99) use_bias_correction=False',
      max_train_epochs: 50,
      save_every_n_epochs: 5,
      train_batch_size: 6,
      mixed_precision: 'bf16',
      multires_noise_iterations: 6,
      multires_noise_discount: 0.3,
      min_snr_gamma: 5,
      min_snr_gamma_enabled: true,
    },
  },
  silvermoong_noobai_loha: {
    name: "silvermoong's LoHa (NoobAI v-pred)",
    description: "Published Yoneyama Mai style LoRA by silvermoong. NoobAI v-prediction base, LoHa via LyCORIS, Prodigy at LR=1, massive effective batch (24×12=288). Pick checkpoint around epoch 115.",
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      lora_type: 'LoHa',
      network_module: 'lycoris.kohya',
      network_dim: 8,
      network_alpha: 8,
      unet_lr: 1,
      text_encoder_lr: 1,
      lr_scheduler: 'constant',
      optimizer_type: 'Prodigy',
      optimizer_args: 'decouple=True weight_decay=0.01 use_bias_correction=True d_coef=2.0',
      max_train_epochs: 200,
      save_every_n_epochs: 5,
      train_batch_size: 24,
      gradient_accumulation_steps: 12,
      mixed_precision: 'bf16',
      v_parameterization: true,
      zero_terminal_snr: true,
      scale_v_pred_loss_like_noise_pred: true,
    },
  },
  rgb_glitch_locon_dora: {
    name: "xRikishi's DoRA + ADOPTAO",
    description: "xRikishi's LoCon+DoRA style LoRA. Inverse alpha/dim ratio (alpha > dim), huge effective batch (6×6=36). NOTE: Original used ADOPTAOScheduleFree optimizer — mapped to AdamW until that package is sourced.",
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      lora_type: 'LoCon',
      network_module: 'lycoris.kohya',
      network_dim: 16,
      network_alpha: 32,
      conv_dim: 10,
      conv_alpha: 20,
      unet_lr: 0.0005,
      text_encoder_lr: 0.00005,
      lr_scheduler: 'constant_with_warmup',
      lr_warmup_steps: 16,
      optimizer_type: 'AdamW',
      optimizer_args: 'betas=(0.8,0.998) weight_decay=2e-07',
      max_train_epochs: 4,
      save_every_n_epochs: 1,
      train_batch_size: 6,
      gradient_accumulation_steps: 6,
      mixed_precision: 'bf16',
      multires_noise_iterations: 6,
      multires_noise_discount: 0.1,
    },
  },
  justtnp_pose_dora: {
    name: "JustTNP's Pose DoRA (PonyXL)",
    description: 'PonyXL LoCon+DoRA via LyCORIS. Tiny network (4/2), near-frozen TE, CAME optimizer, 45 images × 2 repeats, keep_tokens: 1.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      lora_type: 'LoCon',
      network_module: 'lycoris.kohya',
      network_dim: 4,
      network_alpha: 2,
      conv_dim: 4,
      conv_alpha: 2,
      unet_lr: 0.00005,
      text_encoder_lr: 0.000001,
      lr_scheduler: 'cosine',
      optimizer_type: 'CAME',
      optimizer_args: 'weight_decay=0.04',
      max_train_epochs: 6,
      save_every_n_epochs: 1,
      train_batch_size: 2,
      mixed_precision: 'bf16',
      noise_offset: 0.0357,
      multires_noise_iterations: 6,
      multires_noise_discount: 0.3,
      min_snr_gamma: 8,
      min_snr_gamma_enabled: true,
      shuffle_caption: true,
      keep_tokens: 1,
      max_token_length: 225,
    },
  },
  ai_character_flux_dora: {
    name: 'AI Character Style (Flux DoRA)',
    description: 'Flux Dev DoRA via LyCORIS — tiny 18-image dataset, 100 epochs, polynomial scheduler, aggressive pyramid noise.',
    config: {
      model_type: 'Flux',
      resolution: 1024,
      lora_type: 'DoRA',
      network_module: 'lycoris.kohya',
      network_dim: 12,
      network_alpha: 12,
      unet_lr: 0.0003,
      text_encoder_lr: 0.0003,
      lr_scheduler: 'polynomial',
      optimizer_type: 'AdamW',
      optimizer_args: 'weight_decay=0.1',
      max_train_epochs: 100,
      save_every_n_epochs: 10,
      train_batch_size: 1,
      mixed_precision: 'bf16',
      multires_noise_iterations: 10,
      multires_noise_discount: 0.1,
      min_snr_gamma: 5,
      min_snr_gamma_enabled: true,
    },
  },
  crunchy_character: {
    name: 'Crunchy Character',
    description: 'Base SDXL character LoRA (2024). Adafactor with fixed-LR mode (scale_parameter=False), frozen TE, 5 epochs, batch 2.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 32,
      network_alpha: 16,
      unet_lr: 0.0005,
      text_encoder_lr: 0,
      lr_scheduler: 'cosine_with_restarts',
      optimizer_type: 'AdaFactor',
      optimizer_args: 'scale_parameter=False relative_step=False warmup_init=False',
      max_train_epochs: 5,
      save_every_n_epochs: 1,
      train_batch_size: 2,
      mixed_precision: 'bf16',
      noise_offset: 0.1,
      multires_noise_iterations: 6,
      multires_noise_discount: 0.3,
      min_snr_gamma: 5,
      min_snr_gamma_enabled: true,
    },
  },
  citrons_pdxl: {
    name: "Citron's PDXL",
    description: 'Pony Diffusion XL style LoRA. Balanced 1:1 dim/alpha, equal TE/UNet LR, low noise offset, warmup steps.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 32,
      network_alpha: 32,
      unet_lr: 0.0001,
      text_encoder_lr: 0.0001,
      lr_scheduler: 'cosine_with_restarts',
      lr_warmup_steps: 78,
      optimizer_type: 'AdamW8bit' as any,
      optimizer_args: 'weight_decay=0.1 betas=(0.9,0.99)',
      max_train_epochs: 10,
      save_every_n_epochs: 1,
      train_batch_size: 1,
      mixed_precision: 'bf16',
      noise_offset: 0.03,
      multires_noise_iterations: 0,
    },
  },
  citrons_illustrious: {
    name: "Citron's Illustrious",
    description: 'Illustrious style LoRA with Adafactor, active TE at 5e-5, and caption shuffling. 305 images × 4 repeats.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 32,
      network_alpha: 16,
      unet_lr: 0.0005,
      text_encoder_lr: 0.00005,
      lr_scheduler: 'cosine_with_restarts',
      optimizer_type: 'AdaFactor',
      max_train_epochs: 10,
      save_every_n_epochs: 1,
      train_batch_size: 1,
      mixed_precision: 'bf16',
      noise_offset: 0.1,
      multires_noise_iterations: 6,
      multires_noise_discount: 0.3,
      max_token_length: 225,
      shuffle_caption: true,
    },
  },
  citrons_anima: {
    name: "Citron's Anima",
    description: "Anima model style LoRA by Citron. Uses lora_anima network, Qwen VAE, unified LR with warmup, no multinoise.",
    config: {
      model_type: 'Anima',
      resolution: 1024,
      network_dim: 24,
      network_alpha: 24,
      network_module: 'networks.lora_anima',
      unet_lr: 0.0001,
      text_encoder_lr: 0,
      lr_scheduler: 'cosine_with_restarts',
      lr_warmup_steps: 100,
      optimizer_type: 'AdamW8bit' as any,
      optimizer_args: 'weight_decay=0.1 betas=(0.9,0.99)',
      max_train_epochs: 10,
      save_every_n_epochs: 1,
      train_batch_size: 1,
      fp8_base: false,
      mixed_precision: 'bf16',
      noise_offset: 0.03,
      multires_noise_iterations: 0,
      timestep_sampling: 'sigmoid',
      sigmoid_scale: 1,
      discrete_flow_shift: 1,
    },
  },
  illustrious_style_baking: {
    name: 'Style Baking',
    description: 'Illustrious style LoRA with frozen text encoder. Run full 20 epochs and pick the best checkpoint.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 32,
      network_alpha: 16,
      unet_lr: 0.0005,
      text_encoder_lr: 0,
      lr_scheduler: 'cosine_with_restarts',
      optimizer_type: 'AdamW8bit' as any,
      max_train_epochs: 20,
      save_every_n_epochs: 1,
      train_batch_size: 1,
      mixed_precision: 'bf16',
      noise_offset: 0.1,
      multires_noise_iterations: 6,
      multires_noise_discount: 0.3,
      min_snr_gamma: 5,
      min_snr_gamma_enabled: true,
    },
  },
  chroma_style_experimental: {
    name: 'Chroma Style (Experimental)',
    description: 'Chroma LoRA style training — very low dim/alpha ratio, fp8 base, frozen text encoder. Uses lora_flux network.',
    config: {
      model_type: 'Chroma',
      resolution: 1024,
      network_dim: 2,
      network_alpha: 16,
      network_module: 'networks.lora_flux',
      unet_lr: 0.0005,
      text_encoder_lr: 0,
      lr_scheduler: 'cosine_with_restarts',
      optimizer_type: 'AdamW8bit' as any,
      optimizer_args: 'weight_decay=0.01 eps=1e-08',
      max_train_epochs: 20,
      save_every_n_epochs: 1,
      train_batch_size: 2,
      fp8_base: true,
      mixed_precision: 'bf16',
      noise_offset: 0.1,
      multires_noise_iterations: 6,
      multires_noise_discount: 0.3,
      min_snr_gamma: 5,
      min_snr_gamma_enabled: true,
      timestep_sampling: 'sigmoid',
      sigmoid_scale: 1,
      discrete_flow_shift: 3,
      guidance_scale: 0,
      apply_t5_attn_mask: true,
    },
  },
  illustrious_experimental: {
    name: 'Experimental Training',
    description: 'Illustrious style with unfrozen text encoder and weight decay. More flexible, proceed with curiosity.',
    config: {
      model_type: 'SDXL',
      resolution: 1024,
      network_dim: 32,
      network_alpha: 16,
      unet_lr: 0.0005,
      text_encoder_lr: 0.00001,
      lr_scheduler: 'cosine_with_restarts',
      optimizer_type: 'AdamW8bit' as any,
      optimizer_args: 'weight_decay=0.1',
      max_train_epochs: 20,
      save_every_n_epochs: 1,
      train_batch_size: 1,
      mixed_precision: 'bf16',
      noise_offset: 0.1,
      multires_noise_iterations: 6,
      multires_noise_discount: 0.3,
      min_snr_gamma: 5,
      min_snr_gamma_enabled: true,
    },
  },
};
