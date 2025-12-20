/**
 * Saving & Checkpointing Configuration Card
 * Save frequency, format, state management
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { NumberFormField, SelectFormField, CheckboxFormField, TextFormField } from '../fields/FormFields';
import type { TrainingConfig } from '@/lib/api';
import { Save } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface SavingCardProps {
  form: UseFormReturn<TrainingConfig>; // 👈 NO Partial
  onSave?: () => void; // 👈 NEW
}

export function SavingCard({ form, onSave }: SavingCardProps) {
  const saveEveryNEpochs = form.watch('save_every_n_epochs');
  const saveEveryNSteps = form.watch('save_every_n_steps');

  return (
    <Card className="border-green-500/30">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Save className="h-5 w-5 text-green-400" />
            Saving & Checkpoints
          </CardTitle>
          <CardDescription>
            When and how to save LoRA checkpoints
          </CardDescription>
        </div>
        {onSave && (
          <Button type="button" size="sm" variant="outline" onClick={onSave} className="gap-1">
            <Save className="h-3 w-3" />
            Save Saving
          </Button>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Save Format */}
        <SelectFormField
          form={form}
          name="save_model_as"
          label="Save Format"
          description="File format for saved LoRA"
          options={[
            { value: 'safetensors', label: 'SafeTensors', description: 'Recommended, safer and faster' },
            { value: 'pt', label: 'PT (Pickle)', description: 'Pickled tensor format for diffusers' },
          ]}
        />

        {/* Output Name */}
        <TextFormField
          form={form}
          name="output_name"
          label="Output Name (Optional)"
          description="Custom name for saved files (default: project_name)"
          placeholder="my_lora_v1"
        />

        {/* Save Frequency */}
        <div className="grid grid-cols-2 gap-4">
          <NumberFormField
            form={form}
            name="save_every_n_epochs"
            label="Save Every N Epochs"
            description="0 = only save at end"
            placeholder="1"
            min={0}
          />

          <NumberFormField
            form={form}
            name="save_every_n_steps"
            label="Save Every N Steps"
            description="0 = use epochs instead"
            placeholder="0"
            min={0}
          />
        </div>

        {saveEveryNEpochs > 0 && saveEveryNSteps > 0 && (
          <Alert>
            <AlertDescription className="text-xs">
              ⚠️ Both epoch and step saving enabled. Steps will take priority.
            </AlertDescription>
          </Alert>
        )}

        {/* Save Last N */}
        <div className="grid grid-cols-2 gap-4">
          <NumberFormField
            form={form}
            name="save_last_n_epochs"
            label="Keep Last N Epochs"
            description="0 = keep all checkpoints"
            placeholder="0"
            min={0}
          />

          <NumberFormField
            form={form}
            name="save_last_n_epochs_state"
            label="Keep Last N States"
            description="Number of training states to keep"
            placeholder="0"
            min={0}
          />
        </div>

        {/* Training State */}
        <div className="space-y-3 pt-4 border-t border-slate-700">
          <p className="text-sm font-semibold text-gray-300">Training State (Resume)</p>

          <CheckboxFormField
            form={form}
            name="save_state"
            label="Save Training State"
            description="Save optimizer state for resuming (larger files)"
          />

          <CheckboxFormField
            form={form}
            name="save_state_on_train_end"
            label="Save State on Train End"
            description="Always save state at the end of training"
          />

          <TextFormField
            form={form}
            name="resume_from_state"
            label="Resume from State"
            description="Path to training state to resume from"
            placeholder="/path/to/state"
          />
        </div>

        {/* Metadata */}
        <CheckboxFormField
          form={form}
          name="no_metadata"
          label="No Metadata"
          description="Don't save metadata with LoRA (smaller file but less info)"
        />

        {/* Sample Generation */}
        <div className="space-y-3 pt-4 border-t border-slate-700">
          <p className="text-sm font-semibold text-gray-300">Sample Generation (Preview)</p>

          <div className="grid grid-cols-2 gap-4">
            <NumberFormField
              form={form}
              name="sample_every_n_epochs"
              label="Sample Every N Epochs"
              description="0 = no sampling"
              placeholder="0"
              min={0}
            />

            <NumberFormField
              form={form}
              name="sample_every_n_steps"
              label="Sample Every N Steps"
              description="0 = use epochs"
              placeholder="0"
              min={0}
            />
          </div>

          <TextFormField
            form={form}
            name="sample_prompts"
            label="Sample Prompts"
            description="Path to file with prompts, or leave empty"
            placeholder="/path/to/prompts.txt"
          />

          <SelectFormField
            form={form}
            name="sample_sampler"
            label="Sampler"
            description="Sampler for generating previews"
            options={[
              { value: 'euler_a', label: 'Euler Ancestral' },
              { value: 'euler', label: 'Euler' },
              { value: 'ddim', label: 'DDIM' },
              { value: 'ddpm', label: 'DDPM' },
              { value: 'pndm', label: 'PNDM' },
              { value: 'lms', label: 'LMS' },
              { value: 'dpm_2', label: 'DPM 2' },
              { value: 'dpm_2_a', label: 'DPM 2 Ancestral' },
            ]}
          />
        </div>
      </CardContent>
    </Card>
  );
}
