/**
 * LoRA Structure Configuration Card
 * Network architecture: type, dimension, alpha, dropout, and type-specific settings.
 * All type visibility and labels adapt to the selected model and LoRA type.
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { NumberFormField, SelectFormField, CheckboxFormField } from '../fields/FormFields';
import type { TrainingConfig } from '@/lib/api';
import { Network } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Save } from 'lucide-react';

interface LoRAStructureCardProps {
  form: UseFormReturn<TrainingConfig>;
  onSave?: () => void;
}

const LORA_TYPE_OPTIONS = [
  {
    group: 'Standard',
    options: [
      { value: 'LoRA',  label: 'LoRA',  description: 'Standard LoRA (recommended)' },
      { value: 'LoCon', label: 'LoCon', description: 'LoRA + convolution layers (better textures)' },
      { value: 'DoRA',  label: 'DoRA',  description: 'Weight-decomposed LoRA' },
    ],
  },
  {
    group: 'LyCORIS Efficient',
    options: [
      { value: 'LoHa',  label: 'LoHa',  description: 'Hadamard product adaptation' },
      { value: 'LoKr',  label: 'LoKr',  description: 'Kronecker product (compact)' },
      { value: 'TLoRA', label: 'TLoRA', description: 'Timestep-dependent LoRA (SVD-orthogonal init)' },
      { value: 'ABBA',  label: 'ABBA',  description: 'Activation-Based Block Adaptation' },
    ],
  },
  {
    group: 'LyCORIS Advanced',
    options: [
      { value: 'DyLoRA',   label: 'DyLoRA',   description: 'Dynamic resizable rank' },
      { value: 'GLoRA',    label: 'GLoRA',    description: 'Generalized LoRA' },
      { value: 'Diag-OFT', label: 'Diag-OFT', description: 'Diagonal Orthogonal Fine-Tuning' },
      { value: 'BOFT',     label: 'BOFT',     description: 'Butterfly Orthogonal Fine-Tuning' },
      { value: 'IA3',      label: 'IA3',      description: '(IA)³ — inhibition/amplification adapter' },
      { value: 'Full',     label: 'Full',     description: 'Full fine-tuning (DreamBooth-style)' },
    ],
  },
];

const TYPE_HINTS: Record<string, string> = {
  Full:     'Full fine-tuning — equivalent to DreamBooth. Rank and alpha are not used.',
  IA3:      'Requires a high learning rate (5e-3 to 1e-2). Standard LoRA learning rates will underfit.',
  DyLoRA:   'Use a high rank (64+). Alpha should be between rank/4 and rank.',
  'Diag-OFT': 'Orthogonal fine-tuning — fundamentally different from LoRA. Keep rank small (4–16).',
  BOFT:     'Orthogonal fine-tuning — fundamentally different from LoRA. Keep rank small (4–16).',
  TLoRA:    'Rank dropout is ignored — timestep-dependent masking provides equivalent regularisation.',
};

function getNetworkModule(loraType: string, modelType: string): string {
  if (loraType === 'LoRA') {
    return modelType === 'Anima' ? 'networks.lora_anima' : 'networks.lora';
  }
  const algoMap: Record<string, string> = {
    LoCon:     'lycoris.kohya  (algo=locon)',
    LoHa:      'lycoris.kohya  (algo=loha)',
    LoKr:      'lycoris.kohya  (algo=lokr)',
    DoRA:      'lycoris.kohya  (algo=lora, weight_decompose=True)',
    TLoRA:     'lycoris.kohya  (algo=tlora)',
    Full:      'lycoris.kohya  (algo=full)',
    IA3:       'lycoris.kohya  (algo=ia3)',
    DyLoRA:    'lycoris.kohya  (algo=dylora)',
    GLoRA:     'lycoris.kohya  (algo=glora)',
    'Diag-OFT':'lycoris.kohya  (algo=diag-oft)',
    BOFT:      'lycoris.kohya  (algo=boft)',
    ABBA:      'lycoris.kohya  (algo=abba)',
  };
  return algoMap[loraType] ?? 'networks.lora';
}

export function LoRAStructureCard({ form, onSave }: LoRAStructureCardProps) {
  const loraType  = form.watch('lora_type');
  const networkDim   = form.watch('network_dim');
  const networkAlpha = form.watch('network_alpha');
  const modelType = form.watch('model_type');

  const hasConvLayers  = ['LoCon', 'LoHa', 'LoKr'].includes(loraType || '');
  const isFull         = loraType === 'Full';
  const isTransformer  = !['SD15', 'SDXL'].includes(modelType || '');

  const typeHint     = TYPE_HINTS[loraType || ''];
  const networkModule = getNetworkModule(loraType || '', modelType || '');

  return (
    <Card className="border-purple-500/30">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5 text-purple-400" />
            LoRA Structure
          </CardTitle>
          <CardDescription>
            Network architecture and size settings
          </CardDescription>
        </div>
        {onSave && (
          <Button type="button" size="sm" variant="outline" onClick={onSave} className="gap-1">
            <Save className="h-3 w-3" />
            Save LoRA
          </Button>
        )}
      </CardHeader>
      <CardContent className="space-y-4">

        <SelectFormField
          form={form}
          name="lora_type"
          label="LoRA Type"
          description="Network variant to use"
          options={LORA_TYPE_OPTIONS}
        />

        <Alert className="bg-slate-800/50 border-slate-700">
          <AlertDescription className="text-xs font-mono">
            {networkModule}
          </AlertDescription>
        </Alert>

        {typeHint && (
          <Alert className="border-yellow-500/40 bg-yellow-500/5">
            <AlertDescription className="text-xs">
              {typeHint}
            </AlertDescription>
          </Alert>
        )}

        {!isFull && (
          <div className="grid grid-cols-2 gap-4">
            <NumberFormField
              form={form}
              name="network_dim"
              label="Network Dimension (Rank)"
              description="Higher = more detail, larger file"
              placeholder="32"
              min={1}
              max={1024}
            />
            <NumberFormField
              form={form}
              name="network_alpha"
              label="Network Alpha"
              description="Usually equal to dimension"
              placeholder="32"
              min={1}
              max={1024}
            />
          </div>
        )}

        {!isFull && networkDim && networkAlpha && networkDim !== networkAlpha && (
          <Alert>
            <AlertDescription className="text-xs">
              Tip: Alpha is usually set equal to dimension ({networkDim})
            </AlertDescription>
          </Alert>
        )}

        {hasConvLayers && (
          <div className="space-y-4 p-4 border border-purple-500/30 rounded-lg bg-purple-500/5">
            <p className="text-sm font-semibold text-purple-400">
              {loraType} Conv Settings
            </p>

            <div className="grid grid-cols-2 gap-4">
              <NumberFormField
                form={form}
                name="conv_dim"
                label="Conv Dimension"
                description="Rank for convolutional layers"
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
                description="-1 = auto. Higher = more compact"
                placeholder="-1"
                min={-1}
              />
            )}

            <div className="grid grid-cols-2 gap-4">
              <NumberFormField
                form={form}
                name="rank_dropout"
                label="Rank Dropout"
                description="Dropout for rank dimensions (0–1)"
                placeholder="0"
                min={0}
                max={1}
                step={0.01}
              />
              <NumberFormField
                form={form}
                name="module_dropout"
                label="Module Dropout"
                description="Dropout per module (0–1)"
                placeholder="0"
                min={0}
                max={1}
                step={0.01}
              />
            </div>
          </div>
        )}

        <NumberFormField
          form={form}
          name="network_dropout"
          label="Network Dropout"
          description="Dropout rate for LoRA layers (0 = disabled)"
          placeholder="0"
          min={0}
          max={1}
          step={0.01}
        />

        <div className="space-y-3 pt-4 border-t border-slate-700">
          <p className="text-sm font-semibold text-gray-300">Training Target</p>

          <CheckboxFormField
            form={form}
            name="network_train_unet_only"
            label={isTransformer ? 'Train Transformer Only' : 'Train UNet Only'}
            description={isTransformer
              ? 'Skip text encoder training (train the diffusion transformer only)'
              : "Don't train text encoder (recommended for SDXL)"
            }
          />
        </div>

        <CheckboxFormField
          form={form}
          name="dim_from_weights"
          label="Derive Dimension from Weights"
          description="Auto-detect rank from pretrained weights (advanced)"
        />

      </CardContent>
    </Card>
  );
}
