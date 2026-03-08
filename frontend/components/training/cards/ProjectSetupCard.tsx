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
import { Button } from '@/components/ui/button';
import { Save } from 'lucide-react';

// --- Define the props the Card now accepts ---
interface ProjectSetupCardProps {
  form: UseFormReturn<TrainingConfig>; // 👈 NO Partial
  models: { value: string; label: string }[];
  vaes: { value: string; label: string }[];
  onSave?: () => void; // 👈 NEW
}

// --- Update the function signature ---
export function ProjectSetupCard({ form, models, vaes, onSave }: ProjectSetupCardProps) {
  const modelType = form.watch('model_type');
  const needsFluxPaths = modelType === 'Flux' || modelType === 'SD3' || modelType === 'SD3.5';
  const isChroma = modelType === 'Chroma';
  const isAnima = modelType === 'Anima';
  const isHunyuanImage = modelType === 'HunyuanImage';


  return (
    <Card className="border-purple-500/30">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-purple-400" />
            Project Setup
          </CardTitle>
          <CardDescription>
            Basic project information and model selection
          </CardDescription>
        </div>
        {onSave && (
          <Button type="button" size="sm" variant="outline" onClick={onSave} className="gap-1">
            <Save className="h-3 w-3" />
            Save Project
          </Button>
        )}
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
            {
              value: 'Chroma',
              label: 'Chroma',
              description: 'Flux variant (no CLIP-L needed)',
            },
            {
              value: 'Anima',
              label: 'Anima',
              description: 'Qwen3 + T5 dual encoder',
            },
            {
              value: 'HunyuanImage',
              label: 'HunyuanImage',
              description: 'Qwen2.5-VL + byT5 (LoRA only)',
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

        {/* Conditional: Chroma specific paths (T5 + AE, no CLIP-L) */}
        {isChroma && (
          <div className="space-y-4 p-4 border border-green-500/30 rounded-lg bg-green-500/5">
            <div className="flex items-center gap-2 text-sm font-semibold text-green-400">
              <Folder className="h-4 w-4" />
              Chroma Specific Paths
            </div>

            <TextFormField
              form={form}
              name="t5xxl_path"
              label="T5-XXL Path"
              description="Required for Chroma"
              placeholder="/path/to/t5xxl.safetensors"
            />

            <TextFormField
              form={form}
              name="ae_path"
              label="AutoEncoder Path"
              description="Required for Chroma"
              placeholder="/path/to/ae.safetensors"
            />
          </div>
        )}

        {/* Conditional: Anima specific paths (Qwen3 + T5 tokenizer + AE) */}
        {isAnima && (
          <div className="space-y-4 p-4 border border-orange-500/30 rounded-lg bg-orange-500/5">
            <div className="flex items-center gap-2 text-sm font-semibold text-orange-400">
              <Folder className="h-4 w-4" />
              Anima Specific Paths
            </div>

            <TextFormField
              form={form}
              name="qwen3"
              label="Qwen3-0.6B Path"
              description="Required - safetensors file or directory"
              placeholder="/path/to/qwen3-0.6b.safetensors"
            />

            <TextFormField
              form={form}
              name="ae_path"
              label="AutoEncoder (VAE) Path"
              description="Required for Anima"
              placeholder="/path/to/ae.safetensors"
            />

            <TextFormField
              form={form}
              name="t5_tokenizer_path"
              label="T5 Tokenizer Path (Optional)"
              description="Uses default configs/t5_old/ if not set"
              placeholder="/path/to/t5_tokenizer/"
            />

            <TextFormField
              form={form}
              name="llm_adapter_path"
              label="LLM Adapter Path (Optional)"
              description="Separate LLM adapter weights"
              placeholder="/path/to/llm_adapter.safetensors"
            />
          </div>
        )}

        {/* Conditional: HunyuanImage specific paths (Qwen2.5-VL + byT5) */}
        {isHunyuanImage && (
          <div className="space-y-4 p-4 border border-cyan-500/30 rounded-lg bg-cyan-500/5">
            <div className="flex items-center gap-2 text-sm font-semibold text-cyan-400">
              <Folder className="h-4 w-4" />
              HunyuanImage Specific Paths
            </div>

            <TextFormField
              form={form}
              name="text_encoder_path"
              label="Qwen2.5-VL Text Encoder Path"
              description="Required - bfloat16 safetensors"
              placeholder="/path/to/qwen2.5-vl.safetensors"
            />

            <TextFormField
              form={form}
              name="byt5_path"
              label="byT5 Path"
              description="Required - float16 safetensors"
              placeholder="/path/to/byt5.safetensors"
            />
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
