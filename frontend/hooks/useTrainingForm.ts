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
