/**
 * Comprehensive training form hook
 * Combines Zustand persistence + React Hook Form + Zod validation
 * Provides the ultimate UX-friendly form experience
 */

import { useForm, UseFormReturn } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useCallback, useEffect, useRef } from 'react';
import { useTrainingStore } from '@/store/trainingStore';
import { TrainingConfigSchema } from '@/lib/validation';
import type { TrainingConfig } from '@/lib/api';

export function useTrainingForm(options: any = {}) {
  const zustandConfig = useTrainingStore((state) => state.config);
  const updateZustandStore = useTrainingStore((state) => state.updateConfig);
  const loadZustandConfig = useTrainingStore((state) => state.loadConfig);
  const hasHydrated = useTrainingStore((state) => state.hasHydrated);
  const setHasHydrated = useTrainingStore((state) => state.setHasHydrated);

  const form = useForm<TrainingConfig>({
    resolver: zodResolver(TrainingConfigSchema) as any,
    defaultValues: zustandConfig,
    shouldUnregister: false,
  });

  const hasInitializedFromStorage = useRef(false);
  const formRef = useRef(form);
  formRef.current = form;

  // Hydrate form from localStorage ONCE on client
  useEffect(() => {
    if (typeof window === 'undefined') return;

    if (!hasInitializedFromStorage.current) {
      const stored = localStorage.getItem('training-config');
      if (stored) {
        try {
          const { state } = JSON.parse(stored);
          if (state?.config) {
            formRef.current.reset(state.config, { keepDefaultValues: false });
            console.log('✅ Hydrated form from localStorage:', state.config.project_name);
          }
        } catch (e) {
          console.error('Failed to hydrate from localStorage:', e);
        }
      }
      hasInitializedFromStorage.current = true;
      setHasHydrated(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setHasHydrated]); // Only run once

  // ✅ INTENTIONAL SYNC: Call this to push current form data to Zustand
  const syncToStore = useCallback(() => {
    const values = form.getValues();
    updateZustandStore(values);
    console.log("🛰️ Form synced to Zustand for project:", values.project_name);
  }, [form, updateZustandStore]);

  const loadPreset = useCallback((preset: Partial<TrainingConfig>) => {
    if (!preset || typeof preset !== 'object') {
      console.error('❌ Invalid preset:', preset);
      return;
    }

    // Get current config directly from store to avoid stale closure
    const currentConfig = useTrainingStore.getState().config;
    const fullConfig = { ...currentConfig, ...preset } as TrainingConfig;

    console.log('📦 Loading preset:', {
      presetKeys: Object.keys(preset),
      sampleValues: { optimizer: preset.optimizer_type, lr: preset.unet_lr, batch: preset.train_batch_size }
    });

    // Update Zustand store
    loadZustandConfig(fullConfig);

    // Force form to update with new values
    form.reset(fullConfig, { keepDefaultValues: false, keepDirty: false, keepValues: false });

    console.log('✅ Preset loaded, form reset with', fullConfig.optimizer_type);
  }, [loadZustandConfig, form]);

    // ✅ 2. Added this function back in so the UI can show the error list
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
        updateZustandStore(data); // Final sync before training
        callback(data);
      });
    },
    [form, updateZustandStore]
  );

  return {
    form: form as unknown as UseFormReturn<TrainingConfig>,
    config: zustandConfig,
    isDirty: form.formState.isDirty,
    isValid: form.formState.isValid,
    syncToStore, // ✅ Give the UI the ability to save manually
    loadPreset,
    onSubmit,
    getValidationErrors, // 👈 ADD THIS LINE
    resetForm: () => form.reset(zustandConfig),
  };
}

/**
 * Helper hook for form field registration with error display
 * Simplifies common pattern of registering field + showing errors
 */
export function useFormField(
  name: keyof TrainingConfig,
  form: UseFormReturn<TrainingConfig>
) {
  const register = form.register(name);
  // Get error message if it exists
  const error = (form.formState.errors as any)[name]?.message;

  return {
    register,
    error,
    hasError: !!error,
  };
}

/**
 * Configuration presets for common training scenarios
 * We use Partial here because presets don't need project names or paths
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
};
