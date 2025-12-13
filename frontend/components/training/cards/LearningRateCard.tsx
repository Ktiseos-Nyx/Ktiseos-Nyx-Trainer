/**
 * Learning Rate Configuration Card
 * LR settings, scheduler, warmup
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { NumberFormField, SelectFormField } from '../fields/FormFields';
import type { TrainingConfig } from '@/lib/api';
import { TrendingUp } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface LearningRateCardProps {
  form: UseFormReturn<Partial<TrainingConfig>>;
}

export function LearningRateCard({ form }: LearningRateCardProps) {
  const scheduler = form.watch('lr_scheduler');
  const unetLR = form.watch('unet_lr');
  const teLR = form.watch('text_encoder_lr');

  return (
    <Card className="border-orange-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-orange-400" />
          Learning Rates
        </CardTitle>
        <CardDescription>
          Control how fast the model learns
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* UNet Learning Rate */}
        <NumberFormField
          form={form}
          name="unet_lr"
          label="UNet Learning Rate"
          description="Main model learning rate. Typical: 0.0001 - 0.001"
          placeholder="0.0001"
          min={0}
          max={1}
          step={0.00001}
        />

        {/* Text Encoder Learning Rate */}
        <NumberFormField
          form={form}
          name="text_encoder_lr"
          label="Text Encoder Learning Rate"
          description="Usually lower than UNet LR. Set to 0 to freeze text encoder"
          placeholder="0.00005"
          min={0}
          max={1}
          step={0.00001}
        />

        {unetLR && teLR && teLR >= unetLR && (
          <Alert>
            <AlertDescription className="text-xs">
              💡 Tip: Text encoder LR is usually 50-100% of UNet LR
            </AlertDescription>
          </Alert>
        )}

        {/* LR Scheduler */}
        <SelectFormField
          form={form}
          name="lr_scheduler"
          label="Learning Rate Scheduler"
          description="How LR changes during training"
          options={[
            { value: 'constant', label: 'Constant', description: 'Same LR throughout' },
            { value: 'constant_with_warmup', label: 'Constant with Warmup', description: 'Gradual start, then constant' },
            { value: 'linear', label: 'Linear', description: 'Linear decay to zero' },
            { value: 'cosine', label: 'Cosine', description: 'Smooth decay (recommended)' },
            { value: 'cosine_with_restarts', label: 'Cosine with Restarts', description: 'Periodic resets (best for most)' },
            { value: 'polynomial', label: 'Polynomial', description: 'Polynomial decay' },
            { value: 'adafactor', label: 'AdaFactor', description: 'Adaptive LR (use with AdaFactor optimizer)' },
          ]}
        />

        {/* Scheduler-specific settings */}
        {(scheduler === 'cosine_with_restarts' || scheduler === 'polynomial') && (
          <NumberFormField
            form={form}
            name="lr_scheduler_number"
            label={scheduler === 'cosine_with_restarts' ? 'Number of Restarts' : 'Polynomial Degree'}
            description={scheduler === 'cosine_with_restarts' ? 'How many times to restart (typical: 1-3)' : 'Degree of polynomial (typical: 1-3)'}
            placeholder="3"
            min={0}
            max={100}
          />
        )}

        {scheduler === 'polynomial' && (
          <NumberFormField
            form={form}
            name="lr_power"
            label="LR Power"
            description="Power for polynomial scheduler (1.0 = linear)"
            placeholder="1.0"
            min={0}
            max={10}
            step={0.1}
          />
        )}

        {/* Warmup */}
        <div className="space-y-4 p-4 border border-orange-500/30 rounded-lg bg-orange-500/5">
          <p className="text-sm font-semibold text-orange-400">Warmup Settings</p>
          <p className="text-xs text-gray-400">
            Gradually increase LR at the start to stabilize training
          </p>

          <NumberFormField
            form={form}
            name="lr_warmup_ratio"
            label="Warmup Ratio"
            description="Percentage of total steps for warmup (typical: 0.05-0.1)"
            placeholder="0.05"
            min={0}
            max={1}
            step={0.01}
          />

          <NumberFormField
            form={form}
            name="lr_warmup_steps"
            label="Warmup Steps (Optional)"
            description="If > 0, overrides warmup ratio. 0 = use ratio"
            placeholder="0"
            min={0}
          />
        </div>
      </CardContent>
    </Card>
  );
}
