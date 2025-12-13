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

import { useState } from 'react';
import { Play, Save, RotateCcw, AlertCircle, CheckCircle2, Info } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Form } from '@/components/ui/form';
import { useTrainingForm } from '@/hooks/useTrainingForm';
import PresetManager from './PresetManager';
import { trainingAPI } from '@/lib/api';
import {
  SetupTab,
  DatasetTab,
  LoRATab,
  LearningTab,
  AdvancedTab,
  SavingTab,
} from './tabs';

export default function TrainingConfigNew() {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);

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

  /**
   * Start training with validated configuration
   */
  const startTraining = async (validatedConfig: any) => {
    try {
      setIsTraining(true);
      const response = await trainingAPI.start(validatedConfig);

      if (response.success) {
        setTrainingJobId(response.training_id || null);
        alert(`✅ Training started! Job ID: ${response.training_id}`);

        // Redirect to monitor page
        if (response.training_id) {
          window.location.href = `/training/monitor?job=${response.training_id}`;
        }
      } else {
        alert(`❌ Training failed: ${response.message}`);
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
                  <TabsList className="grid w-full grid-cols-6">
                    <TabsTrigger value="setup">Setup</TabsTrigger>
                    <TabsTrigger value="dataset">Dataset</TabsTrigger>
                    <TabsTrigger value="lora">LoRA</TabsTrigger>
                    <TabsTrigger value="learning">Learning</TabsTrigger>
                    <TabsTrigger value="advanced">Advanced</TabsTrigger>
                    <TabsTrigger value="saving">Saving</TabsTrigger>
                  </TabsList>

                  <TabsContent value="setup">
                    <SetupTab form={form} />
                  </TabsContent>

                  <TabsContent value="dataset">
                    <DatasetTab form={form} />
                  </TabsContent>

                  <TabsContent value="lora">
                    <LoRATab form={form} />
                  </TabsContent>

                  <TabsContent value="learning">
                    <LearningTab form={form} />
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
