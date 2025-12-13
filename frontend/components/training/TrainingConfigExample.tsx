/**
 * Example Training Configuration Component
 * Demonstrates how to use the new Zustand + RHF + Zod system
 *
 * This is a simplified reference implementation showing the pattern
 * for refactoring the existing TrainingConfig.tsx component.
 *
 * Key improvements over old approach:
 * 1. State persists across page refreshes (Zustand)
 * 2. Real-time validation with helpful error messages (Zod)
 * 3. Auto-save every 500ms (no more lost configs!)
 * 4. Built-in preset system (SDXL Character, SD1.5, Flux, etc.)
 * 5. Much cleaner code with useTrainingForm hook
 */

'use client';

import { useState } from 'react';
import { Play, Save, RotateCcw, AlertCircle, CheckCircle2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { useTrainingForm } from '@/hooks/useTrainingForm';
import PresetManager from './PresetManager';
import { trainingAPI } from '@/lib/api';

/**
 * Form Field Component with built-in validation display
 * Wraps Input with error handling
 */
interface FormFieldProps {
  label: string;
  name: string;
  type?: string;
  placeholder?: string;
  error?: string;
  register: any;
  helpText?: string;
}

function FormField({ label, name, type = 'text', placeholder, error, register, helpText }: FormFieldProps) {
  return (
    <div className="space-y-2">
      <Label htmlFor={name} className={error ? 'text-red-400' : ''}>
        {label}
        {error && <span className="ml-2 text-xs">⚠️</span>}
      </Label>
      <Input
        id={name}
        type={type}
        placeholder={placeholder}
        {...register}
        className={error ? 'border-red-500 focus-visible:ring-red-500' : ''}
      />
      {error && (
        <p className="text-xs text-red-400 flex items-center gap-1">
          <AlertCircle className="h-3 w-3" />
          {error}
        </p>
      )}
      {helpText && !error && (
        <p className="text-xs text-gray-500">{helpText}</p>
      )}
    </div>
  );
}

/**
 * Example Training Configuration Component
 * Shows how to use the new form system
 */
export default function TrainingConfigExample() {
  const [isTraining, setIsTraining] = useState(false);

  // Initialize the form with all the magic: Zustand + RHF + Zod
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
      alert(`✅ Training started! Job ID: ${response.training_id}`);
    } catch (error: any) {
      alert(`❌ Failed to start training: ${error.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const validationErrors = getValidationErrors();

  return (
    <div className="container mx-auto p-8 max-w-6xl">
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
                Unsaved Changes
              </Badge>
            )}
            {isValid ? (
              <Badge variant="outline" className="border-green-500 text-green-400">
                <CheckCircle2 className="h-3 w-3 mr-1" />
                Valid
              </Badge>
            ) : (
              <Badge variant="outline" className="border-red-500 text-red-400">
                <AlertCircle className="h-3 w-3 mr-1" />
                {validationErrors.length} Error{validationErrors.length !== 1 ? 's' : ''}
              </Badge>
            )}
          </div>
        </div>

        {/* Validation Errors Alert */}
        {validationErrors.length > 0 && (
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
                {validationErrors.length > 5 && (
                  <li className="text-sm text-gray-400">
                    ...and {validationErrors.length - 5} more
                  </li>
                )}
              </ul>
            </AlertDescription>
          </Alert>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Configuration Form */}
          <div className="lg:col-span-2">
            <form onSubmit={onSubmit(startTraining)}>
              <Tabs defaultValue="setup" className="space-y-4">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="setup">Setup</TabsTrigger>
                  <TabsTrigger value="dataset">Dataset</TabsTrigger>
                  <TabsTrigger value="lora">LoRA</TabsTrigger>
                  <TabsTrigger value="learning">Learning</TabsTrigger>
                </TabsList>

                {/* Setup Tab */}
                <TabsContent value="setup" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Project Setup</CardTitle>
                      <CardDescription>Basic project information</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <FormField
                        label="Project Name"
                        name="project_name"
                        placeholder="my_awesome_lora"
                        error={form.formState.errors.project_name?.message}
                        register={form.register('project_name')}
                        helpText="Alphanumeric characters, underscores, and hyphens only"
                      />

                      <div className="space-y-2">
                        <Label>Model Type</Label>
                        <Select
                          value={config.model_type}
                          onValueChange={(value) => form.setValue('model_type', value)}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="SDXL">SDXL</SelectItem>
                            <SelectItem value="SD1.5">SD 1.5</SelectItem>
                            <SelectItem value="Flux">Flux (Experimental)</SelectItem>
                            <SelectItem value="SD3">SD 3</SelectItem>
                            <SelectItem value="SD3.5">SD 3.5</SelectItem>
                            <SelectItem value="Lumina">Lumina</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <FormField
                        label="Base Model Path"
                        name="pretrained_model_name_or_path"
                        placeholder="/path/to/model.safetensors"
                        error={form.formState.errors.pretrained_model_name_or_path?.message}
                        register={form.register('pretrained_model_name_or_path')}
                        helpText="Path to .safetensors or .ckpt file, or HuggingFace model ID"
                      />
                    </CardContent>
                  </Card>
                </TabsContent>

                {/* Dataset Tab */}
                <TabsContent value="dataset" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Dataset Configuration</CardTitle>
                      <CardDescription>Training data settings</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <FormField
                        label="Training Data Directory"
                        name="train_data_dir"
                        placeholder="datasets/my_character"
                        error={form.formState.errors.train_data_dir?.message}
                        register={form.register('train_data_dir')}
                      />

                      <FormField
                        label="Output Directory"
                        name="output_dir"
                        placeholder="output/my_lora"
                        error={form.formState.errors.output_dir?.message}
                        register={form.register('output_dir')}
                      />

                      <FormField
                        label="Resolution"
                        name="resolution"
                        type="number"
                        placeholder="1024"
                        error={form.formState.errors.resolution?.message}
                        register={form.register('resolution', { valueAsNumber: true })}
                        helpText="Must be divisible by 64 (e.g., 512, 768, 1024)"
                      />

                      <FormField
                        label="Batch Size"
                        name="train_batch_size"
                        type="number"
                        placeholder="1"
                        error={form.formState.errors.train_batch_size?.message}
                        register={form.register('train_batch_size', { valueAsNumber: true })}
                        helpText="Lower = less VRAM but slower training"
                      />
                    </CardContent>
                  </Card>
                </TabsContent>

                {/* LoRA Tab */}
                <TabsContent value="lora" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>LoRA Structure</CardTitle>
                      <CardDescription>Network architecture settings</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <FormField
                        label="Network Dimension (Rank)"
                        name="network_dim"
                        type="number"
                        placeholder="32"
                        error={form.formState.errors.network_dim?.message}
                        register={form.register('network_dim', { valueAsNumber: true })}
                        helpText="Higher = more detail but larger file size (16-128)"
                      />

                      <FormField
                        label="Network Alpha"
                        name="network_alpha"
                        type="number"
                        placeholder="32"
                        error={form.formState.errors.network_alpha?.message}
                        register={form.register('network_alpha', { valueAsNumber: true })}
                        helpText="Usually same as dimension"
                      />
                    </CardContent>
                  </Card>
                </TabsContent>

                {/* Learning Rate Tab */}
                <TabsContent value="learning" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Learning Rates</CardTitle>
                      <CardDescription>Training speed settings</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <FormField
                        label="UNet Learning Rate"
                        name="unet_lr"
                        type="number"
                        step="0.00001"
                        placeholder="0.0001"
                        error={form.formState.errors.unet_lr?.message}
                        register={form.register('unet_lr', { valueAsNumber: true })}
                        helpText="Typical: 0.0001 - 0.001"
                      />

                      <FormField
                        label="Text Encoder Learning Rate"
                        name="text_encoder_lr"
                        type="number"
                        step="0.00001"
                        placeholder="0.00005"
                        error={form.formState.errors.text_encoder_lr?.message}
                        register={form.register('text_encoder_lr', { valueAsNumber: true })}
                        helpText="Usually lower than UNet LR"
                      />
                    </CardContent>
                  </Card>
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
                  {isTraining ? 'Starting...' : 'Start Training'}
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
            </form>
          </div>

          {/* Sidebar: Preset Manager */}
          <div className="space-y-6">
            <PresetManager
              currentConfig={config}
              onLoadPreset={loadPreset}
            />

            {/* Quick Stats */}
            <Card className="bg-slate-900/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-sm">Quick Stats</CardTitle>
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
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
