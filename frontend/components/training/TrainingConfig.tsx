'use client';

import { useState, useEffect, useRef } from 'react';
import { Play, Save, RotateCcw, AlertCircle, CheckCircle2, Info } from 'lucide-react';
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

export default function TrainingConfigNew() {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);
  const [models, setModels] = useState<{ value: string; label: string }[]>([]);
  const [vaes, setVaes] = useState<{ value: string; label: string }[]>([]);
  const [datasets, setDatasets] = useState<{ value: string; label: string }[]>([]);
  const [workspaceRoot, setWorkspaceRoot] = useState<string>('');

  // Track initialization to prevent default-wipe loop
  const hasInitialized = useRef(false);

  const {
    form,
    config,
    isDirty,
    isValid,
    syncToStore,
    loadPreset,
    getValidationErrors,
    onSubmit,
    resetForm,
  } = useTrainingForm({
    autoSave: true,
    autoSaveDelay: 500,
    validateOnChange: true,
  });

  useEffect(() => {
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

        // Try datasets then dataset
        let datasetsPath = `${root}/datasets`;
        let datasetsRes = await fetch(`/api/files/list?path=${encodeURIComponent(datasetsPath)}`);
        if (!datasetsRes.ok) {
             datasetsPath = `${root}/dataset`;
             datasetsRes = await fetch(`/api/files/list?path=${encodeURIComponent(datasetsPath)}`);
        }
        const datasetsData = datasetsRes.ok ? await datasetsRes.json() : { files: [] };
        setDatasets((datasetsData.files || []).filter((f: any) => f.type === 'dir').map((dir: any) => ({ value: dir.path, label: dir.name })));

        // Set defaults ONCE
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
  }, [form]);

  const startTraining = async (validatedConfig: any) => {
    try {
      setIsTraining(true);
      const response = await trainingAPI.start(validatedConfig);
      if (response.success) {
        setTrainingJobId(response.job_id || null);
        alert(`✅ Training started! Job ID: ${response.job_id}`);
      } else {
        alert(`❌ Training failed: ${response.message}`);
      }
    } catch (error: any) {
      alert(`❌ Failed to start training: ${error.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const handleSaveToServer = async () => {
    try {
      syncToStore(); // Use the sync function from the hook
      const currentValues = form.getValues();
      const result = await configAPI.save(
        currentValues.project_name || 'untitled_config',
        currentValues as any
      );

      if (result.success) {
        alert("✅ Configuration saved to server!");
      }
    } catch (error: any) {
      alert("❌ Error saving to server: " + error.message);
    }
  };

  const validationErrors = getValidationErrors();
  const errorCount = validationErrors.length;

  const handleSaveDataset = () => {
  syncToStore();
  // Optional: show toast
  // toast.success("Dataset settings saved!");
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

                  <TabsContent value="setup"><SetupTab form={form as any} models={models} vaes={vaes} /></TabsContent>
                  <TabsContent value="dataset"><DatasetTab form={form as any} datasets={datasets} /></TabsContent>
                  <TabsContent value="lora"><LoRATab form={form as any} /></TabsContent>
                  <TabsContent value="learning"><LearningTab form={form as any} /></TabsContent>
                  <TabsContent value="performance"><PerformanceTab form={form as any} /></TabsContent>
                  <TabsContent value="advanced"><AdvancedTab form={form as any} /></TabsContent>
                  <TabsContent value="saving"><SavingTab form={form as any} /></TabsContent>
                  <TabsContent value="learning"><LearningTab form={form} onSave={() => syncToStore()} /></TabsContent>
                </Tabs>

                <div className="mt-6 flex gap-4">
                  <Button type="submit" disabled={!isValid || isTraining} className="bg-linear-to-r from-purple-600 to-pink-600">
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
