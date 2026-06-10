/**
 * Model Settings Card — model-specific tuning.
 *
 * Swaps its body by the selected model family: WD14 confidence thresholds,
 * BLIP sampling/length, or GIT length. Only the active model's controls render.
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Sliders, Sparkles, Camera } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { CheckboxFormField, SliderFormField } from '@/components/training/fields/FormFields';
import type { TaggingConfig } from '@/lib/api';
import { getModelType } from '@/components/tagging/models';

interface ModelSettingsCardProps {
  form: UseFormReturn<TaggingConfig>;
}

export function ModelSettingsCard({ form }: ModelSettingsCardProps) {
  const modelType = getModelType(form.watch('model'));

  if (modelType === 'blip') {
    return (
      <Card className="border-cyan-500/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-cyan-400" />
            BLIP Settings
          </CardTitle>
          <CardDescription>Natural-language caption generation</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <CheckboxFormField
            form={form}
            name="blipBeamSearch"
            label="Beam Search"
            description="Slower but higher quality than nucleus sampling"
          />
          {form.watch('blipBeamSearch') ? (
            <SliderFormField
              form={form}
              name="blipNumBeams"
              label="Number of Beams"
              description="More beams = better quality, slower"
              min={1}
              max={10}
              step={1}
            />
          ) : (
            <SliderFormField form={form} name="blipTopP" label="Top-P (Nucleus)" min={0.5} max={1} step={0.05} />
          )}
          <SliderFormField form={form} name="blipMaxLength" label="Max Length" min={20} max={150} step={5} />
          <SliderFormField form={form} name="blipMinLength" label="Min Length" min={1} max={20} step={1} />
        </CardContent>
      </Card>
    );
  }

  if (modelType === 'git') {
    return (
      <Card className="border-cyan-500/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Camera className="h-5 w-5 text-cyan-400" />
            GIT Settings
          </CardTitle>
          <CardDescription>Photo-style caption generation</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <SliderFormField form={form} name="gitMaxLength" label="Max Caption Length" min={20} max={100} step={5} />
          <CheckboxFormField
            form={form}
            name="gitRemoveWords"
            label='Remove "with the words xxx" artifacts'
          />
        </CardContent>
      </Card>
    );
  }

  // WD14 thresholds
  const useGeneral = form.watch('useGeneralThreshold');
  const useCharacter = form.watch('useCharacterThreshold');
  return (
    <Card className="border-cyan-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sliders className="h-5 w-5 text-cyan-400" />
          Thresholds
        </CardTitle>
        <CardDescription>Confidence cutoffs — lower keeps more tags, higher keeps fewer</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <SliderFormField form={form} name="threshold" label="Overall Threshold" min={0.1} max={0.9} step={0.05} />
        <CheckboxFormField
          form={form}
          name="useGeneralThreshold"
          label="Custom General Threshold"
          description="Override the overall threshold for general tags"
        />
        <SliderFormField
          form={form}
          name="generalThreshold"
          label="General Threshold"
          min={0.1}
          max={0.9}
          step={0.05}
          disabled={!useGeneral}
        />
        <CheckboxFormField
          form={form}
          name="useCharacterThreshold"
          label="Custom Character Threshold"
          description="Override the overall threshold for character tags"
        />
        <SliderFormField
          form={form}
          name="characterThreshold"
          label="Character Threshold"
          min={0.1}
          max={0.9}
          step={0.05}
          disabled={!useCharacter}
        />
      </CardContent>
    </Card>
  );
}
