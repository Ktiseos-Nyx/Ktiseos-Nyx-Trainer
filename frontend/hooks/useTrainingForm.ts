/**
 * Comprehensive training form hook
 * Combines Zustand persistence + React Hook Form + Zod validation
 * Provides the ultimate UX-friendly form experience
 */

import { useForm, UseFormReturn } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useEffect, useCallback } from 'react';
import { useTrainingStore } from '@/store/trainingStore';
import { TrainingConfigSchema } from '@/lib/validation';
import type { TrainingConfig } from '@/lib/api';

export function useTrainingForm(options: any = {}) {
  const { autoSave = true, autoSaveDelay = 500 } = options;

  const zustandConfig = useTrainingStore((state) => state.config);
  const hasHydrated = useTrainingStore((state) => state.hasHydrated);
  const updateZustandConfig = useTrainingStore((state) => state.updateConfig);
  const loadZustandConfig = useTrainingStore((state) => state.loadConfig);

  const form = useForm<TrainingConfig>({
    resolver: zodResolver(TrainingConfigSchema) as any,
    // We only use the store data ONCE on initialization
    defaultValues: zustandConfig,
    shouldUnregister: false,
  });

  // ✅ AUTO-SAVE: ONLY pushes to store. NEVER pulls from store.
  useEffect(() => {
    if (!autoSave || !hasHydrated) return;

    // eslint-disable-next-line react-hooks/incompatible-library
    const subscription = form.watch((value) => {
      // Small debounce
      const timeoutId = setTimeout(() => {
      updateZustandConfig(value as any);
      }, autoSaveDelay);
      return () => clearTimeout(timeoutId);
    });

    return () => subscription.unsubscribe();
  }, [form, autoSave, hasHydrated, autoSaveDelay, updateZustandConfig]);

  // REMOVED: The useEffect that called form.reset() - THIS WAS THE BUG.

  const loadPreset = useCallback((preset: any) => {
    const full = { ...zustandConfig, ...preset };
    loadZustandConfig(full);
    form.reset(full);
  }, [form, loadZustandConfig, zustandConfig]);

  return {
    form: form as any,
    config: zustandConfig,
    isDirty: form.formState.isDirty,
    isValid: form.formState.isValid,
    saveConfig: () => updateZustandConfig(form.getValues()),
    loadPreset,
    onSubmit: (cb: any) => form.handleSubmit(cb),
    resetForm: () => form.reset(zustandConfig),
    getValidationErrors: () => [], // Simplified for now
  };
}

// ✅ THESE MUST BE OUTSIDE THE BRACE ABOVE
export function useFormField(
  name: keyof TrainingConfig,
  form: UseFormReturn<TrainingConfig>
) {
  const register = form.register(name);
  const error = form.formState.errors[name]?.message;

  return {
    register,
    error,
    hasError: !!error,
  };
}

export const trainingPresets: Record<string, { name: string; description: string; config: Partial<TrainingConfig> }> = {
  sdxl_character: {
    name: 'SDXL Character',
    description: 'Optimized for character training',
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
      optimizer_type: 'Prodigy',
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
      optimizer_type: 'AdamW8bit',
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
      optimizer_type: 'AdamW8bit',
      max_train_epochs: 5,
      train_batch_size: 1,
      guidance_scale: 3.5,
      timestep_sampling: 'flux_shift',
    },
  },
};
