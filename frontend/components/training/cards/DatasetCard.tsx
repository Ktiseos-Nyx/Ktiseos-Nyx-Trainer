/**
 * Dataset Configuration Card
 * Essential settings: data paths, resolution, batch size, epochs/steps
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
// Make sure ComboboxFormField is imported
import { TextFormField, NumberFormField, ComboboxFormField } from '../fields/FormFields';
import type { TrainingConfig } from '@/lib/api';
import { Database, Image } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

// --- THIS IS NEW: Define the props this component accepts ---
interface DatasetCardProps {
  form: UseFormReturn<TrainingConfig>; // 👈 NO Partial
  datasets: { value: string; label: string }[];
  onSave?: () => void; // 👈 NEW
}


// --- DELETED THE ROGUE TextFormFieldProps INTERFACE ---

export function DatasetCard({ form, datasets }: DatasetCardProps) {
  const resolution = form.watch('resolution');
  const batchSize = form.watch('train_batch_size');
  const maxEpochs = form.watch('max_train_epochs');
  const maxSteps = form.watch('max_train_steps');

  const estimatedVRAM = resolution === 1024 ? (batchSize || 1) * 8 : (batchSize || 1) * 4;

  return (
    <Card className="border-blue-500/30">
<CardHeader className="flex flex-row items-center justify-between">
  <div>
    <CardTitle className="flex items-center gap-2">
      <Database className="h-5 w-5 text-blue-400" />
      Dataset Configuration
    </CardTitle>
    <CardDescription>
      Training data paths and basic training parameters
    </CardDescription>
  </div>
  {onSave && (
    <Button type="button" size="sm" variant="outline" onClick={onSave} className="gap-1">
      <Save className="h-3 w-3" />
      Save Dataset
    </Button>
  )}
</CardHeader>
      <CardContent className="space-y-4">
        {/* REPLACEMENT FOR TRAINING DATA DIRECTORY */}
        <ComboboxFormField
          form={form}
          name="train_data_dir"
          label="Training Data Directory"
          description="Select a dataset from your /dataset folder"
          placeholder="Select a dataset..."
          options={datasets}
        />

        {/* Make the output directory read-only for clarity */}
        <TextFormField
          form={form}
          name="output_dir"
          label="Output Directory (Default)"
          description="Trained LoRA files will be saved here"
          placeholder="output/my_lora"
          readOnly={true} // <-- Changed to TRUE to make it actually read-only
        />

        {/* Resolution */}
        <NumberFormField
          form={form}
          name="resolution"
          label="Training Resolution"
          description="Must be divisible by 64. SDXL: 1024, SD1.5: 512"
          placeholder="1024"
          min={256}
          max={4096}
          step={64}
        />

        {/* Batch Size */}
        <div className="space-y-2">
          <NumberFormField
            form={form}
            name="train_batch_size"
            label="Batch Size"
            description={`Lower = less VRAM. Est. VRAM: ~${estimatedVRAM}GB`}
            placeholder="1"
            min={1}
            max={128}
          />

          {estimatedVRAM > 24 && (
            <Alert variant="destructive">
              <AlertDescription>
                ⚠️ This batch size may exceed your VRAM! Consider reducing to 1-2.
              </AlertDescription>
            </Alert>
          )}
        </div>

        {/* Repeats */}
        <NumberFormField
          form={form}
          name="num_repeats"
          label="Dataset Repeats"
          description="Higher = more training on same images. Typical: 10-20"
          placeholder="10"
          min={1}
          max={1000}
        />

        {/* Training Duration */}
        <div className="grid grid-cols-2 gap-4">
          <NumberFormField
            form={form}
            name="max_train_epochs"
            label="Max Epochs"
            description="Number of times to train on full dataset"
            placeholder="10"
            min={0}
            max={10000}
          />

          <NumberFormField
            form={form}
            name="max_train_steps"
            label="Max Steps (Optional)"
            description="If > 0, overrides epochs. 0 = use epochs"
            placeholder="0"
            min={0}
          />
        </div>

        {maxSteps > 0 && maxEpochs > 0 && (
          <Alert>
            <AlertDescription>
              <Image className="h-4 w-4 inline mr-2" />
              Max steps ({maxSteps}) will override max epochs ({maxEpochs})
            </AlertDescription>
          </Alert>
        )}

        {/* Seed */}
        <NumberFormField
          form={form}
          name="seed"
          label="Random Seed"
          description="For reproducible results. Use same seed for consistent training"
          placeholder="42"
          min={0}
        />
      </CardContent>
    </Card>
  );
}
