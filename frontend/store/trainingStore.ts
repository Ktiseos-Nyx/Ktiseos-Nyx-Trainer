/**
 * Zustand store for training configuration with localStorage persistence
 * Prevents users from losing their training settings on page refresh
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { useEffect } from 'react';
import type { TrainingConfig } from '@/lib/api';

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
  conv_dim: 32,
  conv_alpha: 32,
  network_dropout: 0.0,
  dim_from_weights: false,
  factor: -1,
  train_norm: false,
  rank_dropout: 0.0,
  module_dropout: 0.0,
  optimizer_type: 'AdamW8bit',
  weight_decay: 0.1,
  gradient_accumulation_steps: 1,
  max_grad_norm: 1.0,
  keep_tokens: 0,
  clip_skip: 2,
  max_token_length: 225,
  caption_dropout_rate: 0.0,
  caption_tag_dropout_rate: 0.0,
  caption_dropout_every_n_epochs: 0,
  keep_tokens_separator: ',',
  secondary_separator: ';',
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
  min_snr_gamma: 0,
  scale_v_pred_loss_like_noise_pred: false,
  v_pred_like_loss: 0,
  debiased_estimation_loss: false,
  noise_offset: 0,
  adaptive_noise_scale: 0,
  ip_noise_gamma: 0,
  ip_noise_gamma_random_strength: false,
  multires_noise_iterations: 0,
  multires_noise_discount: 0.3,
  lowram: false,
  max_data_loader_n_workers: 8,
  persistent_data_loader_workers: 0,
  vae_batch_size: 0,
  min_timestep: 0,
  max_timestep: 1000,
  // ✅ ADD THIS LINE:
  ae_path: '',
};

interface TrainingStore {
  config: TrainingConfig; // Strict!
  isDirty: boolean;
  hasHydrated: boolean; // ✅ ADD THIS
  updateConfig: (updates: Partial<TrainingConfig>) => void; // Partial is okay here
  resetConfig: () => void;
  loadConfig: (config: TrainingConfig) => void; // Strict!
  setDirty: (dirty: boolean) => void;
  getConfig: () => TrainingConfig;
  isValid: () => boolean;
}


export const useTrainingStore = create<TrainingStore>()(
  persist(
    (set, get) => ({
      // Initial state
      config: defaultConfig,
      isDirty: false,
      hasHydrated: false,
      setHasHydrated: (state) => set({ hasHydrated: state }),
      onRehydrateStorage: () => (state) => {
    state?.setHasHydrated(true);
  },

      updateConfig: (updates) => {
        set((state) => ({
          config: { ...state.config, ...updates },
          isDirty: true,
        }));
      },

      resetConfig: () => {
        set({
          config: defaultConfig,
          isDirty: false,
        });
      },

      loadConfig: (config) => {
        set({
          config: { ...defaultConfig, ...config },
          isDirty: false,
        });
      },

      setDirty: (dirty) => {
        set({ isDirty: dirty });
      },

      getConfig: () => get().config,

      isValid: () => {
        const config = get().config;
        return !!(
          config.project_name &&
          config.pretrained_model_name_or_path &&
          config.train_data_dir &&
          config.output_dir
        );
      },
    }),
    {
      name: 'training-config',

      // ✅ FIX 1: Safe storage wrapper (Handles Server vs Client)
      storage: createJSONStorage(() => {
        if (typeof window !== 'undefined') {
          return localStorage;
        }
        return {
          getItem: () => null,
          setItem: () => {},
          removeItem: () => {},
        };
      }),

      // ✅ FIX 2: Skip hydration on server
      skipHydration: true,

      partialize: (state) => ({ config: state.config }),
    }
  )
);

export const useUnsavedChanges = () => {
  const isDirty = useTrainingStore((state) => state.isDirty);

  // ✅ FIX 3: Wrap window access in useEffect
  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.onbeforeunload = isDirty
        ? () => 'You have unsaved training configuration changes. Are you sure you want to leave?'
        : null;
    }
    return () => {
      if (typeof window !== 'undefined') {
        window.onbeforeunload = null;
      }
    };
  }, [isDirty]);

  return isDirty;
};
