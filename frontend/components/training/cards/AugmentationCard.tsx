/**
 * Data Augmentation Configuration Card
 * Settings for image augmentation during training
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { CheckboxFormField, NumberFormField } from '../fields/FormFields';
import type { TrainingConfig } from '@/lib/api';
import { Shuffle } from 'lucide-react';

interface AugmentationCardProps {
  form: UseFormReturn<Partial<TrainingConfig>>;
}

export function AugmentationCard({ form }: AugmentationCardProps) {
  const enableBucket = form.watch('enable_bucket');

  return (
    <Card className="border-green-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Shuffle className="h-5 w-5 text-green-400" />
          Data Augmentation & Bucketing
        </CardTitle>
        <CardDescription>
          Improve training variety and handle multiple resolutions
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Augmentation Options */}
        <div className="space-y-3">
          <p className="text-sm font-semibold text-gray-300">Image Augmentation</p>

          <CheckboxFormField
            form={form}
            name="flip_aug"
            label="Horizontal Flip"
            description="Randomly flip images horizontally during training"
          />

          <CheckboxFormField
            form={form}
            name="random_crop"
            label="Random Crop"
            description="Randomly crop images instead of center crop"
          />

          <CheckboxFormField
            form={form}
            name="color_aug"
            label="Color Augmentation"
            description="Randomly adjust brightness, contrast, saturation"
          />

          <CheckboxFormField
            form={form}
            name="shuffle_caption"
            label="Shuffle Caption Tags"
            description="Randomly reorder caption tokens (helps prevent overfitting)"
          />
        </div>

        {/* Bucketing */}
        <div className="space-y-3 pt-4 border-t border-slate-700">
          <p className="text-sm font-semibold text-gray-300">Resolution Bucketing</p>

          <CheckboxFormField
            form={form}
            name="enable_bucket"
            label="Enable Bucketing"
            description="Allow training on images with different aspect ratios"
          />

          {enableBucket && (
            <div className="space-y-4 pl-6 border-l-2 border-green-500/30">
              <div className="grid grid-cols-2 gap-4">
                <NumberFormField
                  form={form}
                  name="min_bucket_reso"
                  label="Min Resolution"
                  description="Minimum bucket size"
                  placeholder="256"
                  min={64}
                  max={2048}
                  step={64}
                />

                <NumberFormField
                  form={form}
                  name="max_bucket_reso"
                  label="Max Resolution"
                  description="Maximum bucket size"
                  placeholder="2048"
                  min={256}
                  max={4096}
                  step={64}
                />
              </div>

              <NumberFormField
                form={form}
                name="bucket_reso_steps"
                label="Bucket Steps"
                description="Resolution step size for buckets"
                placeholder="64"
                min={8}
                max={128}
                step={8}
              />

              <CheckboxFormField
                form={form}
                name="bucket_no_upscale"
                label="No Upscale"
                description="Don't upscale images smaller than bucket resolution"
              />
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
