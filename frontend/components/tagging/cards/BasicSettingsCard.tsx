/**
 * Basic Settings Card — dataset, model, caption extension/separator.
 */

'use client';

import { useEffect } from 'react';
import { UseFormReturn } from 'react-hook-form';
import Link from 'next/link';
import { Zap } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { SelectFormField, TextFormField } from '@/components/training/fields/FormFields';
import type { DatasetInfo, TaggingConfig } from '@/lib/api';
import { AVAILABLE_MODELS, getModelType } from '@/components/tagging/models';

interface BasicSettingsCardProps {
  form: UseFormReturn<TaggingConfig>;
  datasets: DatasetInfo[];
  loadingDatasets: boolean;
}

const MODEL_GROUPS = [
  { group: 'WD14 Tagger (Anime Tags)', type: 'wd14' as const },
  { group: 'BLIP (Natural Language)', type: 'blip' as const },
  { group: 'GIT (Photo Captions)', type: 'git' as const },
];

export function BasicSettingsCard({ form, datasets, loadingDatasets }: BasicSettingsCardProps) {
  const modelType = getModelType(form.watch('model'));

  // WD14 Danbooru tags must always use .txt
  useEffect(() => {
    if (modelType === 'wd14') {
      form.setValue('captionExtension', '.txt', { shouldDirty: true });
    }
  }, [modelType, form]);

  const datasetOptions = datasets.map((d) => ({
    value: d.path,
    label: `${d.name} (${d.image_count} images)`,
  }));

  const modelOptions = MODEL_GROUPS.map((g) => ({
    group: g.group,
    options: AVAILABLE_MODELS.filter((m) => m.type === g.type).map((m) => ({
      value: m.id,
      label: m.description ? `${m.name} — ${m.description}` : m.name,
    })),
  }));

  return (
    <Card className="border-pink-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="h-5 w-5 text-pink-400" />
          Basic Settings
        </CardTitle>
        <CardDescription>Pick the dataset to tag and the model to tag it with</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {loadingDatasets ? (
          <p className="text-sm text-muted-foreground">Loading datasets…</p>
        ) : datasets.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No datasets found.{' '}
            <Link href="/dataset" prefetch={false} className="text-pink-600 hover:underline">
              Upload images
            </Link>{' '}
            to get started.
          </p>
        ) : (
          <SelectFormField
            form={form}
            name="datasetDir"
            label="Dataset"
            description="The folder of images to tag"
            options={datasetOptions}
          />
        )}

        <SelectFormField
          form={form}
          name="model"
          label="Model"
          description={
            modelType === 'wd14'
              ? 'Generates booru-style tags (1girl, blue_eyes, smile)'
              : modelType === 'blip'
                ? 'Generates natural-language captions'
                : 'Generates photo-style descriptions'
          }
          options={modelOptions}
        />

        {modelType === 'wd14' ? (
          <div className="space-y-2">
            <div>
              <span className="text-sm font-medium">Caption Format</span>
              <p className="text-sm text-muted-foreground">
                <span className="font-mono">.txt</span> — required for Danbooru tags
              </p>
            </div>
            <TextFormField
              form={form}
              name="captionSeparator"
              label="Tag Separator"
              description="String inserted between tags"
              placeholder=", "
            />
          </div>
        ) : (
          <SelectFormField
            form={form}
            name="captionExtension"
            label="Caption Extension"
            options={[
              { value: '.txt', label: '.txt (Kohya standard)' },
              { value: '.caption', label: '.caption' },
              { value: '.cap', label: '.cap' },
            ]}
          />
        )}
      </CardContent>
    </Card>
  );
}
