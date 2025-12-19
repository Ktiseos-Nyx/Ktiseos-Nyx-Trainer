/**
 * Training Configuration Component (New Architecture)
 * Clean orchestrator using modular cards and tabs
 *
 * This replaces the old 2265-line monolith with a clean, maintainable system:
 * - Zustand state persistence
 * - React Hook Form + Zod validation
 * - Modular card architecture
 * - Preset management
 * - Real-time validation
 *
 * Usage: Replace existing TrainingConfig.tsx with this file
 */

'use client';

import { useState, useEffect } from 'react';
import { Play, Save, RotateCcw, AlertCircle, CheckCircle2, Info } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Form } from '@/components/ui/form';
import { useTrainingForm } from '@/hooks/useTrainingForm';
import PresetManager from './PresetManager';
import { trainingAPI, modelsAPI } from '@/lib/api';
import {
  SetupTab,
  DatasetTab,
  LoRATab,
  LearningTab,
  PerformanceTab,
  AdvancedTab,
  SavingTab,
} from './tabs';

// --- Paths removed in favor of dynamic fetching ---

export default function TrainingConfigNew() {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);
    // --- Add these state variables for your file lists ---
  const [models, setModels] = useState<{ value: string; label: string }[]>([]);
  const [vaes, setVaes] = useState<{ value: string; label: string }[]>([]);
  const [datasets, setDatasets] = useState<{ value: string; label: string }[]>([]);
  const [workspaceRoot, setWorkspaceRoot] = useState<string>('');
  // ---------------------------------------------------

  // Initialize the form with all the magic
  const {
    form,
    config,
    isDirty,
    isValid,
    saveConfig,
    resetForm,
    loadPreset,
    getValidationErrors,
    onSubmit,
  } = useTrainingForm({
    autoSave: true,
    autoSaveDelay: 500,
    validateOnChange: true,
  });

  // --- ADD THIS USEEFFECT BLOCK ---
  useEffect(() => {
    const fetchOptions = async () => {
      try {
        // 1. Get the workspace root path
        const workspaceRes = await fetch('/api/files/default-workspace');
        const workspaceData = workspaceRes.ok ? await workspaceRes.json() : { path: '' };
        const root = workspaceData.path;
        
        if (!root) {
          console.error("Failed to determine workspace root");
          return;
        }
        
        setWorkspaceRoot(root);

        // 2. Fetch Models and VAEs using dedicated API (more robust)
        const modelsData = await modelsAPI.list();

        // 3. Fetch Datasets using file API
        // Try 'datasets' (plural) first, as that's the standard
        let datasetsPath = `${root}/datasets`;
        let datasetsRes = await fetch(`/api/files/list?path=${encodeURIComponent(datasetsPath)}`);
        
        // Fallback to 'dataset' (singular) if plural not found
        if (!datasetsRes.ok) {
             datasetsPath = `${root}/dataset`;
             datasetsRes = await fetch(`/api/files/list?path=${encodeURIComponent(datasetsPath)}`);
        }

        const datasetsData = datasetsRes.ok ? await datasetsRes.json() : { files: [] };

        setModels(
          (modelsData.models || [])
            .map((m: any) => ({ value: m.path, label: m.name }))
        );
        setVaes(
          (modelsData.vaes || [])
            .map((v: any) => ({ value: v.path, label: v.name }))
        );
        setDatasets(
          (datasetsData.files || [])
            .filter((f: any) => f.type === 'dir')
            .map((dir: any) => ({ value: dir.path, label: dir.name }))
        );

        // Set sane defaults only if the form is empty for those fields
        if (!form.getValues('output_dir')) {
            form.setValue('output_dir', `${root}/output`);
        }
        if (!form.getValues('pretrained_model_name_or_path') && modelsData.models?.length > 0) {
            form.setValue('pretrained_model_name_or_path', modelsData.models[0].path);
        }
        if (!form.getValues('vae') && modelsData.vaes?.length > 0) {
            // Optional: don't auto-set VAE if you want user to choose
            // form.setValue('vae', modelsData.vaes[0].path);
        }
        if (!form.getValues('train_data_dir') && datasetsData.files?.length > 0) {
            const defaultDataset = datasetsData.files.find((f: any) => f.type === 'dir');
            if (defaultDataset) form.setValue('train_data_dir', defaultDataset.path);
        }

      } catch (error) {
        console.error("Failed to fetch file options for form:", error);
      }
    };

    fetchOptions();
  }, [form]); // Pass form to the dependency array

  // -----------------------------------
  /**
   * Start training with validated configuration
   */
  const startTraining = async (validatedConfig: any) => {
    try {
      setIsTraining(true);
      const response = await trainingAPI.start(validatedConfig);

      if (response.success) {
        setTrainingJobId(response.job_id || null);
        alert(`✅ Training started! Job ID: ${response.job_id}`);

        // Instead of redirecting, rely on TrainingMonitor to pick up job_id from URL if needed
        // Or pass it via state to a sibling component.
        // For now, simply removing the redirect.
        // if (response.job_id) {
        //   window.location.href = `/training/monitor?job=${response.job_id}`;
        // }
      } else {
        if (response.validation_errors && response.validation_errors.length > 0) {
          const details = response.validation_errors
            .map((e: any) => `• ${e.field}: ${e.message}`)
            .join('\n');
          alert(`❌ Validation Failed:\n${response.message}\n\n${details}`);
        } else {
          alert(`❌ Training failed: ${response.message}`);
        }
      }
    } catch (error: any) {
      console.error('Training error:', error);
      alert(`❌ Failed to start training: ${error.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const validationErrors = getValidationErrors();
  const errorCount = validationErrors.length;

  return (
    <div className="container mx-auto p-8 max-w-7xl">
      <Form {...form}>
        <form onSubmit={onSubmit(startTraining)}>
          <div className="space-y-6">
            {/* Header with Status Indicators */}
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
                  Training Configuration
                </h1>
                <p className="text-gray-400 mt-2">
                  Configure your LoRA training with auto-save and validation
                </p>
              </div>
              <div className="flex gap-2">
                {isDirty && (
                  <Badge variant="outline" className="border-yellow-500 text-yellow-400">
                    Auto-saving...
                  </Badge>
                )}
                {isValid ? (
                  <Badge variant="outline" className="border-green-500 text-green-400">
                    <CheckCircle2 className="h-3 w-3 mr-1" />
                    Ready to Train
                  </Badge>
                ) : (
                  <Badge variant="outline" className="border-red-500 text-red-400">
                    <AlertCircle className="h-3 w-3 mr-1" />
                    {errorCount} Error{errorCount !== 1 ? 's' : ''}
                  </Badge>
                )}
              </div>
            </div>

            {/* Validation Errors Alert */}
            {errorCount > 0 && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Configuration Errors</AlertTitle>
                <AlertDescription>
                  <ul className="list-disc list-inside space-y-1 mt-2">
                    {validationErrors.slice(0, 5).map((error, i) => (
                      <li key={i} className="text-sm">
                        <strong>{error.field}:</strong> {error.message}
                      </li>
                    ))}
                    {errorCount > 5 && (
                      <li className="text-sm text-gray-400">
                        ...and {errorCount - 5} more
                      </li>
                    )}
                  </ul>
                </AlertDescription>
              </Alert>
            )}

            {/* Auto-save Notice */}
            <Alert className="bg-blue-500/10 border-blue-500/30">
              <Info className="h-4 w-4" />
              <AlertDescription className="text-sm">
                💾 Your configuration is automatically saved to localStorage. It will survive page refreshes!
              </AlertDescription>
            </Alert>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Main Configuration Tabs */}
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

                  <TabsContent value="setup">
                    <SetupTab form={form} models={models} vaes={vaes} />
                  </TabsContent>

                  <TabsContent value="dataset">
                    <DatasetTab form={form} datasets={datasets} />
                  </TabsContent>

                  <TabsContent value="lora">
                    <LoRATab form={form} />
                  </TabsContent>

                  <TabsContent value="learning">
                    <LearningTab form={form} />
                  </TabsContent>

                  <TabsContent value="performance">
                    <PerformanceTab form={form} />
                  </TabsContent>

                  <TabsContent value="advanced">
                    <AdvancedTab form={form} />
                  </TabsContent>

                  <TabsContent value="saving">
                    <SavingTab form={form} />
                  </TabsContent>
                </Tabs>

                {/* Action Buttons */}
                <div className="mt-6 flex gap-4">
                  <Button
                    type="submit"
                    disabled={!isValid || isTraining}
                    className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
                  >
                    <Play className="h-4 w-4 mr-2" />
                    {isTraining ? 'Starting Training...' : 'Start Training'}
                  </Button>

                  <Button type="button" variant="outline" onClick={saveConfig}>
                    <Save className="h-4 w-4 mr-2" />
                    Save Config
                  </Button>

                  <Button type="button" variant="outline" onClick={resetForm}>
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Reset
                  </Button>
                </div>
              </div>

              {/* Sidebar: Preset Manager + Quick Stats */}
              <div className="space-y-6">
                <PresetManager
                  currentConfig={config}
                  onLoadPreset={loadPreset}
                />

                {/* Quick Stats */}
                <Card className="bg-slate-900/50 border-slate-700">
                  <CardHeader>
                    <CardTitle className="text-sm">Quick Stats</CardTitle>
                    <CardDescription className="text-xs">
                      Current configuration summary
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Model Type:</span>
                      <Badge variant="outline">{config.model_type || 'Not set'}</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Resolution:</span>
                      <span>{config.resolution || 'Not set'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Batch Size:</span>
                      <span>{config.train_batch_size || 'Not set'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Network Dim:</span>
                      <span>{config.network_dim || 'Not set'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">UNet LR:</span>
                      <span className="font-mono text-xs">{config.unet_lr || 'Not set'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Optimizer:</span>
                      <span className="text-xs">{config.optimizer_type || 'Not set'}</span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </form>
      </Form>
    </div>
  );
}
