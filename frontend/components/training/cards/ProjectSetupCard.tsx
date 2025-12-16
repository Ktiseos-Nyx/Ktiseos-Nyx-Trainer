/**
 * Project Setup Configuration Card
 * Contains project name, model type, and base model paths
 *
 * This is an example of the modular card architecture.
 * Each card is ~50-100 lines and handles one logical grouping of settings.
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ComboboxFormField, TextFormField, SelectFormField } from '../fields/FormFields';
import type { TrainingConfig } from '@/lib/api';
import { Folder, Sparkles } from 'lucide-react';

// --- Define the props the Card now accepts ---
interface ProjectSetupCardProps {
  form: UseFormReturn<Partial<TrainingConfig>>;
  models: { value: string; label: string }[];
  vaes: { value: string; label: string }[];
}

// --- Update the function signature ---
export function ProjectSetupCard({ form, models, vaes }: ProjectSetupCardProps) {
  const modelType = form.watch('model_type');
  const needsFluxPaths = modelType === 'Flux' || modelType === 'SD3' || modelType === 'SD3.5';


  return (
    <Card className="border-purple-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-purple-400" />
          Project Setup
        </CardTitle>
        <CardDescription>
          Basic project information and model selection
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Project Name */}
        <TextFormField
          form={form}
          name="project_name"
          label="Project Name"
          description="Alphanumeric characters, underscores, and hyphens only"
          placeholder="my_awesome_lora"
        />

        {/* Model Type Selection */}
        <SelectFormField
          form={form}
          name="model_type"
          label="Model Type"
          description="Choose the base model architecture"
          options={[
            {
              value: 'SDXL',
              label: 'SDXL',
              description: '1024x1024 resolution, most popular',
            },
            {
              value: 'SD1.5',
              label: 'SD 1.5',
              description: '512x512 resolution, classic',
            },
            {
              value: 'Flux',
              label: 'Flux',
              description: 'Experimental, high quality',
            },
            {
              value: 'SD3',
              label: 'SD 3',
              description: 'Latest Stability AI model',
            },
            {
              value: 'SD3.5',
              label: 'SD 3.5',
              description: 'Newest Stability AI release',
            },
            {
              value: 'Lumina',
              label: 'Lumina',
              description: 'Experimental architecture',
            },
          ]}
        />

        {/* --- REPLACEMENT FOR BASE MODEL PATH --- */}
        <ComboboxFormField
          form={form}
          name="pretrained_model_name_or_path"
          label="Base Model Path"
          description="Select a model from your /models/stable-diffusion folder"
          placeholder="Select or type a model path..."
          options={models}
        />

        {/* --- REPLACEMENT FOR VAE PATH --- */}
        <ComboboxFormField
          form={form}
          name="vae_path"
          label="VAE Path (Optional)"
          description="Select a VAE from your /models/vae folder"
          placeholder="Select a VAE..."
          options={vaes}
        />

        {/* Conditional: Flux/SD3 specific paths */}
        {needsFluxPaths && (
          <div className="space-y-4 p-4 border border-blue-500/30 rounded-lg bg-blue-500/5">
            <div className="flex items-center gap-2 text-sm font-semibold text-blue-400">
              <Folder className="h-4 w-4" />
              {modelType} Specific Paths
            </div>

            <TextFormField
              form={form}
              name="clip_l_path"
              label="CLIP-L Path"
              description="Required for Flux/SD3"
              placeholder="/path/to/clip_l.safetensors"
            />

            <TextFormField
              form={form}
              name="clip_g_path"
              label="CLIP-G Path"
              description="Required for Flux/SD3"
              placeholder="/path/to/clip_g.safetensors"
            />

            <TextFormField
              form={form}
              name="t5xxl_path"
              label="T5-XXL Path"
              description="Required for Flux/SD3"
              placeholder="/path/to/t5xxl.safetensors"
            />

            {modelType === 'Flux' && (
              <TextFormField
                form={form}
                name="ae_path"
                label="AutoEncoder Path"
                description="Flux AutoEncoder (*.safetensors)"
                placeholder="/path/to/ae.safetensors"
              />
            )}
          </div>
        )}

        {/* Optional: Continue from existing LoRA */}
        <TextFormField
          form={form}
          name="continue_from_lora"
          label="Continue from LoRA (Optional)"
          description="Path to an existing LoRA to continue training from"
          placeholder="/path/to/existing_lora.safetensors"
        />
      </CardContent>
    </Card>
  );
}
