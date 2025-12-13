/**
 * Caption & Token Configuration Card
 * Token settings, dropout, separators
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { NumberFormField, CheckboxFormField, TextFormField } from '../fields/FormFields';
import type { TrainingConfig } from '@/lib/api';
import { FileText } from 'lucide-react';

interface CaptionCardProps {
  form: UseFormReturn<Partial<TrainingConfig>>;
}

export function CaptionCard({ form }: CaptionCardProps) {
  return (
    <Card className="border-pink-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileText className="h-5 w-5 text-pink-400" />
          Caption & Token Control
        </CardTitle>
        <CardDescription>
          Text processing and caption dropout settings
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Token Settings */}
        <NumberFormField
          form={form}
          name="max_token_length"
          label="Max Token Length"
          description="Maximum caption length. SD1.5: 75, SDXL: 225"
          placeholder="225"
          min={75}
          max={512}
        />

        <NumberFormField
          form={form}
          name="clip_skip"
          label="CLIP Skip"
          description="Skip last N CLIP layers. SD1.5: 1, SDXL: 2"
          placeholder="2"
          min={0}
          max={12}
        />

        <NumberFormField
          form={form}
          name="keep_tokens"
          label="Keep Tokens"
          description="Always keep first N tokens (not subject to dropout)"
          placeholder="0"
          min={0}
        />

        <TextFormField
          form={form}
          name="keep_tokens_separator"
          label="Keep Tokens Separator"
          description="Separator for keep_tokens (e.g., comma)"
          placeholder=","
        />

        {/* Caption Dropout */}
        <div className="space-y-3 pt-4 border-t border-slate-700">
          <p className="text-sm font-semibold text-gray-300">Caption Dropout (Regularization)</p>

          <NumberFormField
            form={form}
            name="caption_dropout_rate"
            label="Caption Dropout Rate"
            description="Probability of dropping entire caption (0-1)"
            placeholder="0"
            min={0}
            max={1}
            step={0.01}
          />

          <NumberFormField
            form={form}
            name="caption_tag_dropout_rate"
            label="Tag Dropout Rate"
            description="Probability of dropping individual tags (0-1)"
            placeholder="0"
            min={0}
            max={1}
            step={0.01}
          />

          <NumberFormField
            form={form}
            name="caption_dropout_every_n_epochs"
            label="Dropout Every N Epochs"
            description="0 = every step, N = only every Nth epoch"
            placeholder="0"
            min={0}
          />
        </div>

        {/* Advanced Caption Settings */}
        <div className="space-y-3 pt-4 border-t border-slate-700">
          <p className="text-sm font-semibold text-gray-300">Advanced</p>

          <TextFormField
            form={form}
            name="secondary_separator"
            label="Secondary Separator"
            description="Secondary separator for captions (e.g., semicolon)"
            placeholder=";"
          />

          <CheckboxFormField
            form={form}
            name="enable_wildcard"
            label="Enable Wildcard"
            description="Enable wildcard expansion in captions"
          />

          <CheckboxFormField
            form={form}
            name="weighted_captions"
            label="Weighted Captions"
            description="Enable weighted captions (e.g., (red hair:1.5))"
          />

          <CheckboxFormField
            form={form}
            name="no_token_padding"
            label="No Token Padding"
            description="Don't pad tokens to max length (memory optimization)"
          />
        </div>
      </CardContent>
    </Card>
  );
}
