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
import type { ModelType, TrainingConfig } from '@/lib/api';

const STORAGE_KEY = 'training-config';

/** Migrate legacy model_type strings to current enum casing. Returns undefined for unknown values. */
export function normalizeModelType(value: string): ModelType | undefined {
  switch (value) {
    case 'SD15':
    case 'SDXL':
    case 'FLUX':
    case 'SD3':
    case 'SD3.5':
    case 'LUMINA':
    case 'Chroma':
    case 'Anima':
    case 'HunyuanImage':
      return value as ModelType;
    case 'SD1.5': return 'SD15';
    case 'Flux':  return 'FLUX';
    case 'Lumina': return 'LUMINA';
    default: return undefined;
  }
}

const SD_ONLY_MODEL_TYPES = new Set<ModelType>(['SD15', 'SDXL']);

/** Strip fields that only apply to SD1.x/SDXL when loading a non-SD config. */
function sanitizeModelFamilyFields(config: Partial<TrainingConfig>): void {
  if (!config.model_type || SD_ONLY_MODEL_TYPES.has(config.model_type)) return;
  delete config.clip_skip;
}

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
  clip_skip: 1,
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
  vae_reflection_padding: false,
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
    merged.model_type = normalizeModelType(merged.model_type) ?? defaultConfig.model_type;
    sanitizeModelFamilyFields(merged);
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

  // 1. Hydrate from localStorage, falling back to server if localStorage is
  //    empty (adblockers, privacy browsers, and fresh cloud sessions wipe it).
  useEffect(() => {
    if (isHydratedRef.current) return;

    const stored = readStoredConfig();
    if (stored) {
      formRef.current.reset(stored, { keepDefaultValues: false });
      console.log('Hydrated form from localStorage:', stored.project_name);
      isHydratedRef.current = true;
      setIsHydrated(true);
    } else {
      // localStorage empty — try server-side last saved form state
      fetch('/api/config/form')
        .then((r) => r.ok ? r.json() : null)
        .then((data) => {
          if (data?.success && data.config) {
            formRef.current.reset(data.config, { keepDefaultValues: false });
            console.log('Hydrated form from server:', data.config.project_name);
          }
        })
        .catch(() => { /* server may not have a saved form yet — that's fine */ })
        .finally(() => {
          isHydratedRef.current = true;
          setIsHydrated(true);
        });
    }
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

    // Normalize legacy model_type strings; fall back to default if unrecognized
    fullConfig.model_type = normalizeModelType(fullConfig.model_type) ?? defaultConfig.model_type;

    // Strip SD-only fields that must not carry into non-SD model families
    sanitizeModelFamilyFields(fullConfig);

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

