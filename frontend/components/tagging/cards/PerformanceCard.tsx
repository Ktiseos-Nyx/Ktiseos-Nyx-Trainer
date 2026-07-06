/**
 * Performance Card — batch/workers, existing-caption handling, runtime flags.
 *
 * Everything is exposed inline (no hidden "Advanced" drawer) — including force-
 * download and debug — so the full control surface is always visible.
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Settings } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { CheckboxFormField, SelectFormField, SliderFormField } from '@/components/training/fields/FormFields';
import type { TaggingConfig } from '@/lib/api';

interface PerformanceCardProps {
  form: UseFormReturn<TaggingConfig>;
}

export function PerformanceCard({ form }: PerformanceCardProps) {
  return (
    <Card className="border-purple-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="h-5 w-5 text-purple-400" />
          Performance
        </CardTitle>
        <CardDescription>Throughput, existing-caption handling, and runtime flags</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <SliderFormField
          form={form}
          name="batchSize"
          label="Batch Size"
          description="Higher = faster (more GPU memory)"
          min={1}
          max={16}
          step={1}
        />
        <SliderFormField form={form} name="maxWorkers" label="Data Loader Workers" min={1} max={8} step={1} />

        <SelectFormField
          form={form}
          name="overwriteMode"
          label="Existing Caption Handling"
          options={[
            { value: 'overwrite', label: 'Overwrite — replace existing captions' },
            { value: 'append', label: 'Append — merge new tags with existing' },
            { value: 'ignore', label: 'Ignore — skip images that already have captions' },
          ]}
        />

        <CheckboxFormField
          form={form}
          name="useOnnx"
          label="Use ONNX Runtime (faster)"
          description="Turn off only on platforms without onnxruntime-gpu (e.g. Apple Silicon / CPU-only)"
        />
        <CheckboxFormField form={form} name="recursive" label="Recursive (process subfolders)" />
        <CheckboxFormField form={form} name="frequencyTags" label="Show Tag Frequency Report" />
        <CheckboxFormField form={form} name="forceDownload" label="Force Download Model" />
        <CheckboxFormField form={form} name="debug" label="Debug Mode" />
      </CardContent>
    </Card>
  );
}
