/**
 * LoRA Structure Configuration Card
 * Network architecture: dimension, alpha, type, dropout
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { NumberFormField, SelectFormField, CheckboxFormField } from '../fields/FormFields';
import type { TrainingConfig } from '@/lib/api';
import { Network } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface LoRAStructureCardProps {
  form: UseFormReturn<Partial<TrainingConfig>>;
}

export function LoRAStructureCard({ form }: LoRAStructureCardProps) {
  const loraType = form.watch('lora_type');
  const networkDim = form.watch('network_dim');
  const networkAlpha = form.watch('network_alpha');

  const isLycoris = ['LoCon', 'LoHa', 'LoKr', 'DoRA'].includes(loraType || '');

  return (
    <Card className="border-purple-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Network className="h-5 w-5 text-purple-400" />
          LoRA Structure
        </CardTitle>
        <CardDescription>
          Network architecture and size settings
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* LoRA Type */}
        <SelectFormField
          form={form}
          name="lora_type"
          label="LoRA Type"
          description="LoRA variant to use"
          options={[
            { value: 'LoRA', label: 'LoRA', description: 'Standard LoRA (recommended)' },
            { value: 'LoCon', label: 'LoCon', description: 'LoRA for convolutions (better textures)' },
            { value: 'LoHa', label: 'LoHa', description: 'Hadamard product adaptation' },
            { value: 'LoKr', label: 'LoKr', description: 'Kronecker product (compact)' },
            { value: 'DoRA', label: 'DoRA', description: 'Weight-decomposed LoRA (experimental)' },
          ]}
        />

        {/* Network Module (auto-set based on type) */}
        <Alert className="bg-slate-800/50 border-slate-700">
          <AlertDescription className="text-xs">
            Network module: {loraType === 'LoRA' ? 'networks.lora' : 'lycoris.kohya'}
          </AlertDescription>
        </Alert>

        {/* Dimension and Alpha */}
        <div className="grid grid-cols-2 gap-4">
          <NumberFormField
            form={form}
            name="network_dim"
            label="Network Dimension (Rank)"
            description="Higher = more detail but larger file"
            placeholder="32"
            min={1}
            max={1024}
          />

          <NumberFormField
            form={form}
            name="network_alpha"
            label="Network Alpha"
            description="Usually same as dimension"
            placeholder="32"
            min={1}
            max={1024}
          />
        </div>

        {networkDim && networkAlpha && networkDim !== networkAlpha && (
          <Alert>
            <AlertDescription className="text-xs">
              💡 Tip: Alpha is usually set equal to dimension ({networkDim})
            </AlertDescription>
          </Alert>
        )}

        {/* Convolution Settings (LoCon/LoHa only) */}
        {isLycoris && (
          <div className="space-y-4 p-4 border border-purple-500/30 rounded-lg bg-purple-500/5">
            <p className="text-sm font-semibold text-purple-400">
              {loraType} Specific Settings
            </p>

            <div className="grid grid-cols-2 gap-4">
              <NumberFormField
                form={form}
                name="conv_dim"
                label="Conv Dimension"
                description="Dimension for convolutional layers"
                placeholder="32"
                min={1}
                max={1024}
              />

              <NumberFormField
                form={form}
                name="conv_alpha"
                label="Conv Alpha"
                description="Alpha for convolutional layers"
                placeholder="32"
                min={1}
                max={1024}
              />
            </div>

            {loraType === 'LoKr' && (
              <NumberFormField
                form={form}
                name="factor"
                label="Decomposition Factor"
                description="-1 = auto-detect. Higher = more compact"
                placeholder="-1"
                min={-1}
              />
            )}

            <CheckboxFormField
              form={form}
              name="train_norm"
              label="Train Normalization Layers"
              description="Include normalization layers in training (LyCORIS)"
            />

            <div className="grid grid-cols-2 gap-4">
              <NumberFormField
                form={form}
                name="rank_dropout"
                label="Rank Dropout"
                description="Dropout for rank dimensions (0-1)"
                placeholder="0"
                min={0}
                max={1}
                step={0.01}
              />

              <NumberFormField
                form={form}
                name="module_dropout"
                label="Module Dropout"
                description="Dropout for modules (0-1)"
                placeholder="0"
                min={0}
                max={1}
                step={0.01}
              />
            </div>
          </div>
        )}

        {/* Network Dropout */}
        <NumberFormField
          form={form}
          name="network_dropout"
          label="Network Dropout"
          description="Dropout rate for LoRA layers (0 = no dropout)"
          placeholder="0"
          min={0}
          max={1}
          step={0.01}
        />

        {/* Advanced: Dimension from weights */}
        <CheckboxFormField
          form={form}
          name="dim_from_weights"
          label="Derive Dimension from Weights"
          description="Auto-detect dimension from pretrained weights (advanced)"
        />
      </CardContent>
    </Card>
  );
}
