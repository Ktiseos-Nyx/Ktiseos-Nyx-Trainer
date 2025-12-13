/**
 * Advanced Training Configuration Card
 * SNR gamma, noise settings, timestep sampling, Flux/Lumina specific
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { NumberFormField, SelectFormField, CheckboxFormField, TextFormField } from '../fields/FormFields';
import type { TrainingConfig } from '@/lib/api';
import { Settings2 } from 'lucide-react';

interface AdvancedCardProps {
  form: UseFormReturn<Partial<TrainingConfig>>;
}

export function AdvancedCard({ form }: AdvancedCardProps) {
  const modelType = form.watch('model_type');
  const isFlux = modelType === 'Flux';
  const isLumina = modelType === 'Lumina';
  const isSD2 = modelType === 'SD2.0' || modelType === 'SD2.1';

  return (
    <Card className="border-red-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings2 className="h-5 w-5 text-red-400" />
          Advanced Settings
        </CardTitle>
        <CardDescription>
          Experimental features and advanced techniques
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* SNR & Noise */}
        <div className="space-y-3">
          <p className="text-sm font-semibold text-gray-300">SNR & Noise Settings</p>

          <NumberFormField
            form={form}
            name="min_snr_gamma"
            label="Min SNR Gamma"
            description="Helps convergence. 0 = disabled, typical: 5.0"
            placeholder="0"
            min={0}
            max={20}
          />

          <NumberFormField
            form={form}
            name="noise_offset"
            label="Noise Offset"
            description="Add noise offset for better dark/light images (0-0.1)"
            placeholder="0"
            min={0}
            max={1}
            step={0.01}
          />

          <NumberFormField
            form={form}
            name="adaptive_noise_scale"
            label="Adaptive Noise Scale"
            description="Adaptive noise scaling (0-1)"
            placeholder="0"
            min={0}
            max={1}
            step={0.01}
          />

          <NumberFormField
            form={form}
            name="ip_noise_gamma"
            label="Input Perturbation Noise"
            description="Adds robustness. Typical: 0.05"
            placeholder="0"
            min={0}
            max={1}
            step={0.01}
          />

          <CheckboxFormField
            form={form}
            name="ip_noise_gamma_random_strength"
            label="Random Strength IP Noise"
            description="Randomize input perturbation strength"
          />

          <NumberFormField
            form={form}
            name="multires_noise_iterations"
            label="Multi-res Noise Iterations"
            description="0 = disabled, typical: 6-10"
            placeholder="0"
            min={0}
          />

          <NumberFormField
            form={form}
            name="multires_noise_discount"
            label="Multi-res Noise Discount"
            description="Discount factor for multi-res noise (0-1)"
            placeholder="0.3"
            min={0}
            max={1}
            step={0.01}
          />

          <CheckboxFormField
            form={form}
            name="zero_terminal_snr"
            label="Zero Terminal SNR"
            description="Force noise to zero at final timestep"
          />
        </div>

        {/* Loss Functions */}
        <div className="space-y-3 pt-4 border-t border-slate-700">
          <p className="text-sm font-semibold text-gray-300">Loss Functions</p>

          <CheckboxFormField
            form={form}
            name="scale_v_pred_loss_like_noise_pred"
            label="Scale V-Pred Loss Like Noise Pred"
            description="Scale v-prediction loss (for v-pred models)"
          />

          <NumberFormField
            form={form}
            name="v_pred_like_loss"
            label="V-Pred Like Loss Weight"
            description="0 = disabled"
            placeholder="0"
            min={0}
          />

          <CheckboxFormField
            form={form}
            name="debiased_estimation_loss"
            label="Debiased Estimation Loss"
            description="Use debiased estimation for loss calculation"
          />

          <NumberFormField
            form={form}
            name="prior_loss_weight"
            label="Prior Loss Weight"
            description="Weight for prior loss (0-1)"
            placeholder="0"
            min={0}
            max={1}
            step={0.01}
          />
        </div>

        {/* Timestep Settings */}
        <div className="space-y-3 pt-4 border-t border-slate-700">
          <p className="text-sm font-semibold text-gray-300">Timestep Settings</p>

          <div className="grid grid-cols-2 gap-4">
            <NumberFormField
              form={form}
              name="min_timestep"
              label="Min Timestep"
              description="Minimum diffusion timestep"
              placeholder="0"
              min={0}
            />

            <NumberFormField
              form={form}
              name="max_timestep"
              label="Max Timestep"
              description="Maximum diffusion timestep"
              placeholder="1000"
              min={0}
            />
          </div>
        </div>

        {/* SD 2.x Specific */}
        {isSD2 && (
          <div className="space-y-3 p-4 border border-blue-500/30 rounded-lg bg-blue-500/5">
            <p className="text-sm font-semibold text-blue-400">SD 2.x Settings</p>

            <CheckboxFormField
              form={form}
              name="v2"
              label="SD 2.x Base Model"
              description="Enable for SD 2.0/2.1 models"
            />

            <CheckboxFormField
              form={form}
              name="v_parameterization"
              label="V-Parameterization"
              description="For SDXL v-pred or SD 2.x 768px models"
            />
          </div>
        )}

        {/* Flux Specific */}
        {isFlux && (
          <div className="space-y-3 p-4 border border-purple-500/30 rounded-lg bg-purple-500/5">
            <p className="text-sm font-semibold text-purple-400">Flux Settings</p>

            <NumberFormField
              form={form}
              name="t5xxl_max_token_length"
              label="T5-XXL Max Token Length"
              description="256 for schnell, 512 for dev"
              placeholder="512"
              min={128}
              max={1024}
            />

            <CheckboxFormField
              form={form}
              name="apply_t5_attn_mask"
              label="Apply T5 Attention Mask"
              description="Apply attention mask to T5-XXL"
            />

            <NumberFormField
              form={form}
              name="guidance_scale"
              label="Guidance Scale"
              description="For Flux.1 dev. Typical: 3.5"
              placeholder="3.5"
              min={0}
              max={30}
              step={0.1}
            />

            <SelectFormField
              form={form}
              name="timestep_sampling"
              label="Timestep Sampling"
              description="Sampling strategy for timesteps"
              options={[
                { value: 'sigma', label: 'Sigma' },
                { value: 'uniform', label: 'Uniform' },
                { value: 'sigmoid', label: 'Sigmoid' },
                { value: 'shift', label: 'Shift' },
                { value: 'flux_shift', label: 'Flux Shift (Recommended)' },
              ]}
            />

            <NumberFormField
              form={form}
              name="sigmoid_scale"
              label="Sigmoid Scale"
              description="Scale for sigmoid timestep sampling"
              placeholder="1.0"
              min={0}
              step={0.1}
            />

            <SelectFormField
              form={form}
              name="model_prediction_type"
              label="Model Prediction Type"
              description="Prediction type for Flux"
              options={[
                { value: 'raw', label: 'Raw' },
                { value: 'additive', label: 'Additive (for dev model)' },
              ]}
            />

            <NumberFormField
              form={form}
              name="blocks_to_swap"
              label="Blocks to Swap"
              description="Number of blocks to swap (memory optimization)"
              placeholder="0"
              min={0}
            />
          </div>
        )}

        {/* Lumina Specific */}
        {isLumina && (
          <div className="space-y-3 p-4 border border-yellow-500/30 rounded-lg bg-yellow-500/5">
            <p className="text-sm font-semibold text-yellow-400">Lumina Settings</p>

            <TextFormField
              form={form}
              name="gemma2"
              label="Gemma2 Model Path"
              description="Path to Gemma2 model (*.sft or *.safetensors)"
              placeholder="/path/to/gemma2.safetensors"
            />

            <NumberFormField
              form={form}
              name="gemma2_max_token_length"
              label="Gemma2 Max Token Length"
              description="Default: 256"
              placeholder="256"
              min={128}
              max={1024}
            />
          </div>
        )}

        {/* General Advanced */}
        <div className="space-y-3 pt-4 border-t border-slate-700">
          <p className="text-sm font-semibold text-gray-300">Other Advanced Options</p>

          <CheckboxFormField
            form={form}
            name="network_train_unet_only"
            label="Train UNet Only"
            description="Don't train text encoder (recommended for SDXL)"
          />

          <CheckboxFormField
            form={form}
            name="lowram"
            label="Low RAM Mode"
            description="Reduce RAM usage at cost of speed"
          />
        </div>
      </CardContent>
    </Card>
  );
}
