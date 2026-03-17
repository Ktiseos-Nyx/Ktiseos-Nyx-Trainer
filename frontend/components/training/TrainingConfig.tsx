'use client';

import { useState, useEffect, useRef } from 'react';
import { Play, Save, RotateCcw, AlertCircle, CheckCircle2, Info } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Form } from '@/components/ui/form';
import { useTrainingForm } from '@/hooks/useTrainingForm';
import PresetManager from './PresetManager';
import { trainingAPI, modelsAPI, configAPI } from '@/lib/api';
import {
  SetupTab,
  DatasetTab,
  LoRATab,
  LearningTab,
  PerformanceTab,
  AdvancedTab,
  SavingTab,
} from './tabs';

/**
 * Main training configuration page component.
 *
 * Renders a tabbed form covering all training settings (Setup, Dataset, LoRA,
 * Learning, Performance, Advanced, Saving), a preset manager panel, a
 * validation error banner, and action buttons for starting training, saving
 * config to disk, and resetting to defaults.
 *
 * Responsibilities:
 * - Waits for localStorage hydration via `useTrainingForm` before fetching
 *   remote data, preventing default-value race conditions.
 * - Fetches available models, VAEs, and datasets from the API post-hydration.
 * - Sets smart per-field defaults on first mount only (`hasInitialized` guard).
 * - Submits the validated config to the training API and tracks the returned job ID.
 * - Delegates per-card saves to `handleCardSave`, which generates both
 *   `dataset.toml` and `config.toml` via `configAPI.saveTraining`.
 *
 * `@returns` The rendered training configuration UI.
 */
export default function TrainingConfigNew() {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);
  const [models, setModels] = useState<{ value: string; label: string }[]>([]);
  const [vaes, setVaes] = useState<{ value: string; label: string }[]>([]);
  const [textEncoders, setTextEncoders] = useState<{ value: string; label: string }[]>([]);
  const [datasets, setDatasets] = useState<{ value: string; label: string }[]>([]);
  const [workspaceRoot, setWorkspaceRoot] = useState<string>('');

  // Track initialization to prevent default-wipe loop
  const hasInitialized = useRef(false);

  const {
    form,
    config,
    isDirty,
    isValid,
    isHydrated,
    syncToStore,
    loadPreset,
    getValidationErrors,
    onSubmit,
    resetForm,
  } = useTrainingForm({
    autoSave: true,
    autoSaveDelay: 500,
    validateOnChange: false,  // 👈 Ensure this is false (or just remove the line)
  });

  // Fetch dropdown options AFTER hydration completes
  useEffect(() => {
    // Wait for localStorage hydration before fetching options
    if (!isHydrated) return;

    const fetchOptions = async () => {
      try {
        const workspaceRes = await fetch('/api/files/default-workspace');
        const workspaceData = workspaceRes.ok ? await workspaceRes.json() : { path: '' };
        const root = workspaceData.path;
        if (!root) return;
        setWorkspaceRoot(root);

        const modelsData = await modelsAPI.list();
        setModels((modelsData.models || []).map((m: any) => ({ value: m.path, label: m.name })));
        setVaes((modelsData.vaes || []).map((v: any) => ({ value: v.path, label: v.name })));
        setTextEncoders((modelsData.text_encoders || []).map((m: any) => ({ value: m.path, label: m.name })));

        // Try datasets then dataset
        let datasetsPath = `${root}/datasets`;
        let datasetsRes = await fetch(`/api/files/list?path=${encodeURIComponent(datasetsPath)}`);
        if (!datasetsRes.ok) {
             datasetsPath = `${root}/dataset`;
             datasetsRes = await fetch(`/api/files/list?path=${encodeURIComponent(datasetsPath)}`);
        }
        const datasetsData = datasetsRes.ok ? await datasetsRes.json() : { files: [] };
        setDatasets((datasetsData.files || []).filter((f: any) => f.type === 'dir').map((dir: any) => ({ value: dir.path, label: dir.name })));

        // Set defaults ONLY for truly empty fields (not overwriting hydrated values)
        if (!hasInitialized.current) {
            if (!form.getValues('output_dir')) form.setValue('output_dir', `${root}/output`);
            if (!form.getValues('pretrained_model_name_or_path') && modelsData.models?.length > 0) {
                form.setValue('pretrained_model_name_or_path', modelsData.models[0].path);
            }
            if (!form.getValues('train_data_dir') && datasetsData.files?.length > 0) {
                const def = datasetsData.files.find((f: any) => f.type === 'dir');
                if (def) form.setValue('train_data_dir', def.path);
            }
            hasInitialized.current = true;
        }
      } catch (error) {
        console.error("Fetch options error:", error);
      }
    };
    fetchOptions();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isHydrated]);

  const startTraining = async (validatedConfig: any) => {
    try {
      setIsTraining(true);
      const response = await trainingAPI.start(validatedConfig);
      if (response.success) {
        setTrainingJobId(response.job_id || null);
        // Store job ID for TrainingMonitor to pick up
        if (response.job_id && typeof window !== 'undefined') {
          localStorage.setItem('current_training_job_id', response.job_id);
          window.dispatchEvent(new CustomEvent('training-started', {
            detail: { jobId: response.job_id },
          }));
        }
        toast.success(`Training started! Job ID: ${response.job_id}`);
      } else {
        // Show validation error details if available
        const errors = response.validation_errors || [];
        const errorDetails = errors
          .filter((e: any) => e.severity === 'error')
          .map((e: any) => `${e.field}: ${e.message}`)
          .join('; ');
        const warnings = errors
          .filter((e: any) => e.severity === 'warning')
          .map((e: any) => `${e.field}: ${e.message}`)
          .join('; ');

        toast.error(`Training failed: ${response.message}`, {
          description: [errorDetails, warnings].filter(Boolean).join(' | ') || undefined,
          duration: 10000,
        });
      }
    } catch (error: any) {
      toast.error(`Failed to start training: ${error.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const handleSaveToServer = async () => {
    try {
      syncToStore(); // Force save to localStorage
      const currentValues = form.getValues();

      // Use new endpoint that generates both dataset.toml and config.toml
      const result = await configAPI.saveTraining(currentValues);

      if (result.success) {
        toast.success('Training configs saved', {
          description: `Files: ${result.files.dataset}, ${result.files.config}`,
        });
      }
    } catch (error: any) {
      toast.error(`Error saving configs: ${error.message}`);
    }
  };

  const validationErrors = getValidationErrors();
  const errorCount = validationErrors.length;

  // Handler for individual card save buttons
  const handleCardSave = async () => {
    try {
      syncToStore(); // Force save to localStorage
      const currentValues = form.getValues();

      // Also save configs to disk (generates both dataset.toml and config.toml)
      await configAPI.saveTraining(currentValues);

      // Optional: You can add toast notification here instead of alert
      console.log("✅ Config saved to disk");
    } catch (error: any) {
      console.error("❌ Error saving config:", error.message);
      // Optionally show error to user
    }
  };

  return (
    <div className="container mx-auto p-8 max-w-7xl">
      <Form {...form}>
        <form onSubmit={onSubmit(startTraining)}>
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold bg-linear-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
                  Training Configuration
                </h1>
              </div>
              <div className="flex gap-2">
                {isDirty && <Badge variant="outline" className="border-yellow-500 text-yellow-400">Unsaved Changes</Badge>}
                {isValid ? (
                  <Badge variant="outline" className="border-green-500 text-green-400"><CheckCircle2 className="h-3 w-3 mr-1" />Ready</Badge>
                ) : (
                  <Badge variant="outline" className="border-red-500 text-red-400"><AlertCircle className="h-3 w-3 mr-1" />{errorCount} Errors</Badge>
                )}
              </div>
            </div>

            {errorCount > 0 && (
              <Alert variant="destructive">
                <AlertTitle>Configuration Errors</AlertTitle>
                <AlertDescription>
                  <ul className="list-disc list-inside space-y-1 mt-2">
                    {validationErrors.slice(0, 5).map((error, i) => (
                      <li key={i} className="text-sm"><strong>{error.field}:</strong> {error.message}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <Tabs defaultValue="setup" className="space-y-4">
                  <TabsList className="grid w-full grid-cols-7">
                    <TabsTrigger value="setup">Setup</TabsTrigger>
                    <TabsTrigger value="dataset">Dataset</TabsTrigger>
                    <TabsTrigger value="lora">LoRA</TabsTrigger>
                    <TabsTrigger value="learning">Learning</TabsTrigger>
                    <TabsTrigger value="performance">Performance</TabsTrigger>
                    <TabsTrigger value="advanced">Advanced</TabsTrigger>
                    <TabsTrigger value="saving">Saving</TabsTrigger>
                  </TabsList>

                  <TabsContent value="setup"><SetupTab form={form as any} models={models} vaes={vaes} textEncoders={textEncoders} onSave={handleCardSave} /></TabsContent>
                  <TabsContent value="dataset"><DatasetTab form={form as any} datasets={datasets} onSave={handleCardSave} /></TabsContent>
                  <TabsContent value="lora"><LoRATab form={form as any} onSave={handleCardSave} /></TabsContent>
                  <TabsContent value="learning"><LearningTab form={form as any} onSave={handleCardSave} /></TabsContent>
                  <TabsContent value="performance"><PerformanceTab form={form as any} onSave={handleCardSave} /></TabsContent>
                  <TabsContent value="advanced"><AdvancedTab form={form as any} onSave={handleCardSave} /></TabsContent>
                  <TabsContent value="saving"><SavingTab form={form as any} onSave={handleCardSave} /></TabsContent>
                </Tabs>

                <div className="mt-6 flex gap-4">
                  <Button type="submit" disabled={isTraining} className="bg-linear-to-r from-purple-600 to-pink-600">
                    <Play className="h-4 w-4 mr-2" />
                    {isTraining ? 'Starting...' : 'Start Training'}
                  </Button>
                  <Button type="button" variant="outline" onClick={handleSaveToServer}>
                    <Save className="h-4 w-4 mr-2" />Save Config
                  </Button>
                  <Button type="button" variant="outline" onClick={resetForm}>
                    <RotateCcw className="h-4 w-4 mr-2" />Reset
                  </Button>
                </div>
              </div>


              <div className="space-y-6">
                <PresetManager currentConfig={config} onLoadPreset={loadPreset} />
              </div>
            </div>
          </div>
        </form>
      </Form>
    </div>
  );
}
