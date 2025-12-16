/**
 * Comprehensive training form hook
 * Combines Zustand persistence + React Hook Form + Zod validation
 * Provides the ultimate UX-friendly form experience
 */

import { useForm, UseFormReturn, FieldError } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useEffect, useCallback } from 'react';
import { useTrainingStore } from '@/store/trainingStore';
import { PartialTrainingConfigSchema, formatValidationErrors } from '@/lib/validation';
import type { TrainingConfig } from '@/lib/api';
import { z } from 'zod';

/**
 * Options for the training form hook
 */
export interface UseTrainingFormOptions {
  /** Auto-save to localStorage on every change */
  autoSave?: boolean;
  /** Debounce time in ms for auto-save (default: 500ms) */
  autoSaveDelay?: number;
  /** Validate on change (default: true) */
  validateOnChange?: boolean;
  /** Validate on blur (default: true) */
  validateOnBlur?: boolean;
}

/**
 * Return type for useTrainingForm hook
 */
export interface UseTrainingFormReturn {
  /** React Hook Form instance */
  form: UseFormReturn<Partial<TrainingConfig>>;
  /** Current configuration from Zustand */
  config: Partial<TrainingConfig>;
  /** Check if form has unsaved changes */
  isDirty: boolean;
  /** Check if form is valid */
  isValid: boolean;
  /** Save configuration to Zustand */
  saveConfig: () => void;
  /** Reset form to Zustand state */
  resetForm: () => void;
  /** Load a preset configuration */
  loadPreset: (preset: Partial<TrainingConfig>) => void;
  /** Get validation errors in user-friendly format */
  getValidationErrors: () => Array<{ field: string; message: string }>;
  /** Submit handler for training start */
  onSubmit: (callback: (config: Partial<TrainingConfig>) => void) => (e?: React.BaseSyntheticEvent) => void;
}

/**
 * Custom hook for training configuration form
 *
 * This hook combines:
 * - Zustand for state persistence (survives page refresh)
 * - React Hook Form for form state management
 * - Zod for schema validation
 *
 * Usage:
 * ```tsx
 * const { form, saveConfig, isValid, onSubmit } = useTrainingForm();
 *
 * // Register a field
 * <Input {...form.register('project_name')} />
 *
 * // Show validation errors
 * {form.formState.errors.project_name && (
 *   <span>{form.formState.errors.project_name.message}</span>
 * )}
 *
 * // Submit
 * <form onSubmit={onSubmit(startTraining)}>
 * ```
 */
export function useTrainingForm(options: UseTrainingFormOptions = {}): UseTrainingFormReturn {
  const {
    autoSave = true,
    autoSaveDelay = 500,
    validateOnChange = true,
    validateOnBlur = true,
  } = options;

  // Get Zustand store state and actions
  const zustandConfig = useTrainingStore((state) => state.config);
  const updateZustandConfig = useTrainingStore((state) => state.updateConfig);
  const resetZustandConfig = useTrainingStore((state) => state.resetConfig);
  const loadZustandConfig = useTrainingStore((state) => state.loadConfig);
  const setDirty = useTrainingStore((state) => state.setDirty);
  const zustandIsValid = useTrainingStore((state) => state.isValid);

  // Initialize React Hook Form with Zod validation
  const form = useForm<Partial<TrainingConfig>>({
    resolver: zodResolver(PartialTrainingConfigSchema),
    mode: validateOnChange ? 'onChange' : 'onBlur',
    defaultValues: zustandConfig,
    reValidateMode: 'onChange',
  });

  // Watch all form values for auto-save
  const formValues = form.watch();

  // Auto-save to Zustand store with debounce
  useEffect(() => {
    if (!autoSave) return;

    const timeoutId = setTimeout(() => {
      // Only save if form has changed
      if (form.formState.isDirty) {
        updateZustandConfig(formValues);
        setDirty(true);
      }
    }, autoSaveDelay);

    return () => clearTimeout(timeoutId);
  }, [formValues, autoSave, autoSaveDelay, form.formState.isDirty, updateZustandConfig, setDirty]);

  // Sync Zustand changes back to form (e.g., when loading a preset)
  useEffect(() => {
    form.reset(zustandConfig, { keepDirty: false });
  }, [zustandConfig]); // Only re-run if Zustand config changes

  /**
   * Save current form values to Zustand store
   */
  const saveConfig = useCallback(() => {
    const values = form.getValues();
    updateZustandConfig(values);
    setDirty(false);
    form.reset(values, { keepDirty: false }); // Reset dirty state
  }, [form, updateZustandConfig, setDirty]);

  /**
   * Reset form to last saved Zustand state
   */
  const resetForm = useCallback(() => {
    form.reset(zustandConfig, { keepDirty: false });
    setDirty(false);
  }, [form, zustandConfig, setDirty]);

  /**
   * Load a preset configuration
   */
  const loadPreset = useCallback((preset: Partial<TrainingConfig>) => {
    loadZustandConfig(preset);
    form.reset(preset, { keepDirty: false });
    setDirty(false);
  }, [form, loadZustandConfig, setDirty]);

  /**
   * Get validation errors in user-friendly format
   */
  const getValidationErrors = useCallback(() => {
    const errors = form.formState.errors;
    return Object.entries(errors).map(([field, error]) => ({
      field,
      // FIX: Cast error as 'any' so we can access .message
      message: (error as any)?.message || 'Invalid value',
    }));
  }, [form.formState.errors]);

  /**
   * Submit handler wrapper
   */
  const onSubmit = useCallback(
    (callback: (config: Partial<TrainingConfig>) => void) => {
      return form.handleSubmit((data) => {
        // Save to Zustand before submitting
        updateZustandConfig(data);
        setDirty(false);
        callback(data);
      });
    },
    [form, updateZustandConfig, setDirty]
  );

  return {
    form,
    config: zustandConfig,
    isDirty: form.formState.isDirty,
    isValid: form.formState.isValid && zustandIsValid(),
    saveConfig,
    resetForm,
    loadPreset,
    getValidationErrors,
    onSubmit,
  };
}

/**
 * Helper hook for form field registration with error display
 * Simplifies common pattern of registering field + showing errors
 *
 * Usage:
 * ```tsx
 * const { register, error } = useFormField('project_name', form);
 *
 * <Input {...register} />
 * {error && <span className="text-red-500">{error}</span>}
 * ```
 */
export function useFormField(
  name: keyof TrainingConfig,
  form: UseFormReturn<Partial<TrainingConfig>>
) {
  const register = form.register(name);
  const error = form.formState.errors[name]?.message;

  return {
    register,
    error,
    hasError: !!error,
  };
}

/**
 * Configuration presets for common training scenarios
 */
export const trainingPresets: Record<string, { name: string; description: string; config: Partial<TrainingConfig> }> = {
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
      optimizer_type: 'AdamW8bit',
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
