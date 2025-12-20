/**
 * Optimizer Configuration Card
 * Optimizer type, weight decay, gradient settings
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { NumberFormField, SelectFormField, TextFormField } from '../fields/FormFields';
import type { TrainingConfig } from '@/lib/api';
import { Zap } from 'lucide-react';

interface OptimizerCardProps {
  form: UseFormReturn<TrainingConfig>; // 👈 NO Partial
  onSave?: () => void; // 👈 NEW
}

export function OptimizerCard({ form, onSave }: OptimizerCardProps) {
  return (
    <Card className="border-yellow-500/30">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-yellow-400" />
            Optimizer Settings
          </CardTitle>
          <CardDescription>
            Optimization algorithm and training stability
          </CardDescription>
        </div>
        {onSave && (
          <Button type="button" size="sm" variant="outline" onClick={onSave} className="gap-1">
            <Save className="h-3 w-3" />
            Save Optimizer
          </Button>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Optimizer Type */}
        <SelectFormField
          form={form}
          name="optimizer_type"
          label="Optimizer"
          description="Algorithm for updating weights"
          options={[
            { value: 'AdamW', label: 'AdamW', description: 'Standard adaptive optimizer' },
            { value: 'AdamW8bit', label: 'AdamW8bit', description: 'Memory-efficient AdamW (recommended)' },
            { value: 'Lion', label: 'Lion', description: 'Newer, more efficient' },
            { value: 'Lion8bit', label: 'Lion8bit', description: 'Memory-efficient Lion' },
            { value: 'Prodigy', label: 'Prodigy', description: 'Auto-adjusting LR (experimental)' },
            { value: 'AdaFactor', label: 'AdaFactor', description: 'Memory-efficient, use with adafactor scheduler' },
            { value: 'DAdaptation', label: 'DAdaptation', description: 'Auto-adjusting (experimental)' },
            { value: 'DAdaptAdam', label: 'DAdaptAdam', description: 'D-adapted Adam' },
            { value: 'DAdaptAdaGrad', label: 'DAdaptAdaGrad', description: 'D-adapted AdaGrad' },
            { value: 'DAdaptAdan', label: 'DAdaptAdan', description: 'D-adapted Adan' },
            { value: 'DAdaptSGD', label: 'DAdaptSGD', description: 'D-adapted SGD' },
            { value: 'SGDNesterov', label: 'SGDNesterov', description: 'Nesterov SGD' },
            { value: 'SGDNesterov8bit', label: 'SGDNesterov8bit', description: 'Memory-efficient Nesterov' },
            { value: 'CAME', label: 'CAME', description: 'Confidence-guided adaptive (experimental)' },
          ]}
        />

        {/* Weight Decay */}
        <NumberFormField
          form={form}
          name="weight_decay"
          label="Weight Decay"
          description="L2 regularization. Typical: 0.01-0.1"
          placeholder="0.01"
          min={0}
          max={1}
          step={0.01}
        />

        {/* Gradient Accumulation */}
        <NumberFormField
          form={form}
          name="gradient_accumulation_steps"
          label="Gradient Accumulation Steps"
          description="Effective batch size = batch_size × accumulation_steps"
          placeholder="1"
          min={1}
          max={128}
        />

        {/* Max Gradient Norm */}
        <NumberFormField
          form={form}
          name="max_grad_norm"
          label="Max Gradient Norm"
          description="Gradient clipping to prevent instability. Typical: 1.0"
          placeholder="1.0"
          min={0}
          max={10}
          step={0.1}
        />

        {/* Optional: Custom Optimizer Args */}
        <TextFormField
          form={form}
          name="optimizer_args"
          label="Optimizer Arguments (Advanced)"
          description="JSON string of custom optimizer arguments (leave empty unless you know what you're doing)"
          placeholder='{"betas": [0.9, 0.999]}'
        />
      </CardContent>
    </Card>
  );
}
