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
  const updateZustandConfig = useTrainingStore((state) => state.updateConfig);
  const loadZustandConfig = useTrainingStore((state) => state.loadConfig);
  const setDirty = useTrainingStore((state) => state.setDirty);

  const form = useForm<TrainingConfig>({
    resolver: zodResolver(TrainingConfigSchema) as any,
    defaultValues: zustandConfig,
    // ✅ Keep data when tabs unmount
    shouldUnregister: false,
  });

  // ✅ FIX: Using 'eslint-disable-line' at the end of the line is often cleaner
  // We need watch() for auto-save, even if the compiler thinks it's "incompatible"
  const formValues = form.watch(); // eslint-disable-line react-hooks/incompatible-library

  // ✅ AUTO-SAVE FIX:
  useEffect(() => {
    if (!autoSave) return;

    const timeoutId = setTimeout(() => {
      // Use getValues() instead of the watched formValues to get the absolute latest data
      if (form.formState.isDirty) {
        const latestData = form.getValues();
        console.log("💾 Persistent Sync:", latestData.project_name);
        updateZustandConfig(latestData);

        // We reset the "dirty" state locally so we don't spam the store,
        // but we keep the current values.
        form.reset(latestData, { keepValues: true, keepDirty: false });
      }
    }, autoSaveDelay);

    return () => clearTimeout(timeoutId);
  }, [formValues, autoSave, autoSaveDelay, updateZustandConfig, form]);

  // ✅ Preset Loader
  const loadPreset = useCallback((preset: Partial<TrainingConfig>) => {
    const fullConfig = { ...zustandConfig, ...preset } as TrainingConfig;
    loadZustandConfig(fullConfig);
    form.reset(fullConfig, { keepDirty: false });
    setDirty(false);
  }, [zustandConfig, loadZustandConfig, form, setDirty]);

  const saveConfig = useCallback(() => {
    const values = form.getValues();
    updateZustandConfig(values);
    setDirty(false);
    form.reset(values, { keepDirty: false });
  }, [form, updateZustandConfig, setDirty]);

  const onSubmit = useCallback(
    (callback: (config: TrainingConfig) => void) => {
      return form.handleSubmit((data) => {
        updateZustandConfig(data);
        setDirty(false);
        callback(data);
      });
    },
    [form, updateZustandConfig, setDirty]
  );

  return {
    form: form as unknown as UseFormReturn<TrainingConfig>,
    config: zustandConfig,
    isDirty: form.formState.isDirty,
    isValid: form.formState.isValid,
    saveConfig,
    loadPreset,
    onSubmit,
    resetForm: () => form.reset(zustandConfig),
    getValidationErrors: () => Object.entries(form.formState.errors).map(([f, e]) => ({
      field: f,
      message: (e as any)?.message || 'Invalid value'
    })),
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
