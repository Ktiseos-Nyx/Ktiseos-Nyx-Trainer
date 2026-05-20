/**
 * Logging Configuration Card
 * Configures TensorBoard and WandB logging for training runs.
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { SelectFormField, TextFormField } from '../fields/FormFields';
import type { TrainingConfig } from '@/lib/api';
import { BarChart2 } from 'lucide-react';

interface LoggingCardProps {
  form: UseFormReturn<TrainingConfig>;
}

export function LoggingCard({ form }: LoggingCardProps) {
  const logWith = form.watch('log_with');
  const showTensorboard = logWith === 'tensorboard' || logWith === 'all';
  const showWandB = logWith === 'wandb' || logWith === 'all';

  return (
    <Card className="border-blue-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart2 className="h-5 w-5 text-blue-400" />
          Logging
        </CardTitle>
        <CardDescription>
          Send training metrics to TensorBoard or WandB
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <SelectFormField
          form={form}
          name="log_with"
          label="Log With"
          description="Where to send training metrics (loss, learning rate, etc.)"
          options={[
            { value: '', label: 'None' },
            { value: 'tensorboard', label: 'TensorBoard' },
            { value: 'wandb', label: 'WandB' },
            { value: 'all', label: 'Both' },
          ]}
        />

        {showTensorboard && (
          <div className="space-y-4 pt-2 border-t border-slate-700">
            <p className="text-sm font-semibold text-gray-300">TensorBoard</p>

            <TextFormField
              form={form}
              name="logging_dir"
              label="Log Directory"
              description="Directory where TensorBoard event files are written"
              placeholder="./logs"
            />

            <TextFormField
              form={form}
              name="log_prefix"
              label="Log Prefix (Optional)"
              description="Prefix for the timestamped log folder (e.g. 'sdxl-' → logs/sdxl-20260520123456)"
              placeholder="my-experiment-"
            />
          </div>
        )}

        {showWandB && (
          <div className="space-y-4 pt-2 border-t border-slate-700">
            <p className="text-sm font-semibold text-gray-300">WandB</p>

            <FormField
              control={form.control}
              name="wandb_key"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>API Key</FormLabel>
                  <FormControl>
                    <Input
                      type="password"
                      placeholder="wbapi-..."
                      {...field}
                      value={field.value ?? ''}
                    />
                  </FormControl>
                  <FormDescription>
                    Your WandB API key —{' '}
                    <a
                      href="https://wandb.ai/authorize"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline text-blue-400"
                    >
                      get it here
                    </a>
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <TextFormField
              form={form}
              name="log_tracker_name"
              label="Project Name (Optional)"
              description="WandB project that groups all your runs (e.g. 'sdxl-experiments')"
              placeholder="my-lora-project"
            />

            <TextFormField
              form={form}
              name="wandb_run_name"
              label="Run Name (Optional)"
              description="Name for this individual training run"
              placeholder="sdxl-v2-attempt-1"
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
