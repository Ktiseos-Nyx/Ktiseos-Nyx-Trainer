/**
 * Memory & Performance Configuration Card
 * Precision, caching, gradient checkpointing
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { NumberFormField, SelectFormField, CheckboxFormField } from '../fields/FormFields';
import type { TrainingConfig } from '@/lib/api';
import { Cpu } from 'lucide-react';

interface MemoryCardProps {
  form: UseFormReturn<TrainingConfig>; // 👈 NO Partial
  onSave?: () => void; // 👈 NEW
}

export function MemoryCard({ form, onSave }: MemoryCardProps) {
  const cacheLatents = form.watch('cache_latents');
  const cacheText = form.watch('cache_text_encoder_outputs');

  return (
    <Card className="border-cyan-500/30">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="h-5 w-5 text-cyan-400" />
            Memory & Performance
          </CardTitle>
          <CardDescription>
            Precision settings and memory optimizations
          </CardDescription>
        </div>
        {onSave && (
          <Button type="button" size="sm" variant="outline" onClick={onSave} className="gap-1">
            <Save className="h-3 w-3" />
            Save Memory
          </Button>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Mixed Precision */}
        <SelectFormField
          form={form}
          name="mixed_precision"
          label="Mixed Precision"
          description="Lower precision = less VRAM, slightly faster"
          options={[
            { value: 'no', label: 'No (FP32)', description: 'Full precision, most VRAM' },
            { value: 'fp16', label: 'FP16', description: 'Half precision (recommended)' },
            { value: 'bf16', label: 'BF16', description: 'Brain float16 (newer GPUs only)' },
          ]}
        />

        {/* Save Precision */}
        <SelectFormField
          form={form}
          name="save_precision"
          label="Save Precision"
          description="Precision for saved LoRA file"
          options={[
            { value: 'fp16', label: 'FP16', description: 'Half precision (smaller file)' },
            { value: 'bf16', label: 'BF16', description: 'Brain float16' },
            { value: 'fp32', label: 'FP32', description: 'Full precision (larger file)' },
          ]}
        />

        {/* Full Precision Options */}
        <div className="space-y-3">
          <p className="text-sm font-semibold text-gray-300">Full Precision Training</p>

          <CheckboxFormField
            form={form}
            name="full_fp16"
            label="Full FP16"
            description="Train entirely in FP16 (saves VRAM but may be unstable)"
          />

          <CheckboxFormField
            form={form}
            name="full_bf16"
            label="Full BF16"
            description="Train entirely in BF16 (newer GPUs)"
          />

          <CheckboxFormField
            form={form}
            name="fp8_base"
            label="FP8 Base Model"
            description="Load base model in FP8 (experimental, saves VRAM)"
          />
        </div>

        {/* Caching */}
        <div className="space-y-3 pt-4 border-t border-slate-700">
          <p className="text-sm font-semibold text-gray-300">Caching (Speed Optimization)</p>

          <CheckboxFormField
            form={form}
            name="cache_latents"
            label="Cache Latents"
            description="Pre-encode images (faster training, less VRAM during training)"
          />

          {cacheLatents && (
            <div className="pl-6 border-l-2 border-cyan-500/30">
              <CheckboxFormField
                form={form}
                name="cache_latents_to_disk"
                label="Cache to Disk"
                description="Save latents to disk (slower but uses less RAM)"
              />
            </div>
          )}

          <CheckboxFormField
            form={form}
            name="cache_text_encoder_outputs"
            label="Cache Text Encoder Outputs"
            description="Pre-encode captions (faster, allows freezing text encoder)"
          />

          {cacheText && (
            <div className="pl-6 border-l-2 border-cyan-500/30">
              <CheckboxFormField
                form={form}
                name="cache_text_encoder_outputs_to_disk"
                label="Cache Text to Disk"
                description="Save text embeddings to disk"
              />
            </div>
          )}
        </div>

        {/* Advanced Memory Settings */}
        <div className="space-y-3 pt-4 border-t border-slate-700">
          <p className="text-sm font-semibold text-gray-300">Advanced Settings</p>

          <CheckboxFormField
            form={form}
            name="gradient_checkpointing"
            label="Gradient Checkpointing"
            description="Trade compute for VRAM (slower but uses less memory)"
          />

          <CheckboxFormField
            form={form}
            name="no_half_vae"
            label="No Half VAE"
            description="Use full precision VAE (if you get NaN errors)"
          />

          <NumberFormField
            form={form}
            name="vae_batch_size"
            label="VAE Batch Size"
            description="Number of images processed by VAE per batch (min 1). Lower = less VRAM."
            placeholder="1"
            min={1}
            max={128}
          />

          <NumberFormField
            form={form}
            name="max_data_loader_n_workers"
            label="Data Loader Workers"
            description="Number of CPU workers for data loading (0 = auto)"
            placeholder="8"
            min={0}
            max={32}
          />

          <CheckboxFormField
            form={form}
            name="persistent_data_loader_workers"
            label="Persistent Data Loader Workers"
            description="Keep workers alive between epochs (faster)"
          />
        </div>

        {/* Memory Optimization */}
        <div className="space-y-3 pt-4 border-t border-slate-700">
          <p className="text-sm font-semibold text-gray-300">Memory Optimization</p>

          <CheckboxFormField
            form={form}
            name="lowram"
            label="Low RAM Mode"
            description="Reduce RAM usage at cost of speed"
          />
        </div>
      </CardContent>
    </Card>
  );
}
