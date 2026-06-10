/**
 * Tag Processing Card (WD14 only) — prefix/blacklist/replacement, rating, flags.
 *
 * For BLIP/GIT this surface doesn't apply, so it renders a short note instead of
 * an empty panel.
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Tag } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { CheckboxFormField, SelectFormField, TextFormField } from '@/components/training/fields/FormFields';
import type { TaggingConfig } from '@/lib/api';
import { getModelType } from '@/components/tagging/models';

interface TagProcessingCardProps {
  form: UseFormReturn<TaggingConfig>;
}

export function TagProcessingCard({ form }: TagProcessingCardProps) {
  if (getModelType(form.watch('model')) !== 'wd14') {
    return (
      <Card className="border-orange-500/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Tag className="h-5 w-5 text-orange-400" />
            Tag Processing
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Tag processing options apply to WD14 tagging only.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-orange-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Tag className="h-5 w-5 text-orange-400" />
          Tag Processing
        </CardTitle>
        <CardDescription>Ordering, filtering, and replacement applied to generated tags</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <TextFormField
          form={form}
          name="alwaysFirstTags"
          label="Prefix Tags (Optional)"
          description="Tags to always put first"
          placeholder="e.g. 1girl, solo"
        />
        <TextFormField
          form={form}
          name="undesiredTags"
          label="Undesired Tags (Optional)"
          description="Comma-separated tags to exclude"
          placeholder="e.g. watermark, logo, text"
        />
        <TextFormField
          form={form}
          name="tagReplacement"
          label="Tag Replacement (Optional)"
          description="Format: source,target;source,target"
          placeholder="old1,new1;old2,new2"
        />
        <SelectFormField
          form={form}
          name="ratingTags"
          label="Rating Tags"
          description="Whether to include the model's content rating, and where"
          options={[
            { value: 'none', label: 'None' },
            { value: 'first', label: 'First' },
            { value: 'last', label: 'Last' },
          ]}
        />
        <CheckboxFormField form={form} name="characterTagsFirst" label="Character Tags First" />
        <CheckboxFormField form={form} name="removeUnderscore" label="Remove Underscores" />
        <CheckboxFormField
          form={form}
          name="characterTagExpand"
          label="Expand Character Tags"
          description="name_(series) → name, series"
        />
      </CardContent>
    </Card>
  );
}
