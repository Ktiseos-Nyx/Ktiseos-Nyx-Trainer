/**
 * Training Configuration Tab Components
 * Each tab composes related configuration cards
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
import { Form } from '@/components/ui/form';
import type { TrainingConfig } from '@/lib/api';
import { ProjectSetupCard } from '../cards/ProjectSetupCard';
import { DatasetCard } from '../cards/DatasetCard';
import { AugmentationCard } from '../cards/AugmentationCard';
import { LoRAStructureCard } from '../cards/LoRAStructureCard';
import { LearningRateCard } from '../cards/LearningRateCard';
import { OptimizerCard } from '../cards/OptimizerCard';
import { MemoryCard } from '../cards/MemoryCard';
import { SavingCard } from '../cards/SavingCard';
import { CaptionCard } from '../cards/CaptionCard';
import { AdvancedCard } from '../cards/AdvancedCard';

// ✅ NEW:
interface TabProps {
  form: UseFormReturn<TrainingConfig>; // Remove Partial here
  models?: { value: string; label: string }[];
  vaes?: { value: string; label: string }[];
  datasets?: { value: string; label: string }[];
	onSave?: () => void; // 👈 ADD THIS
}

/**
 * Setup Tab
 * Project information and model selection
 */
export function SetupTab({ form, models = [], vaes = [], onSave }: TabProps) {
  return (
    <div className="space-y-6">
      <ProjectSetupCard form={form} models={models} vaes={vaes} onSave={onSave} />
    </div>
  );
}

/**
 * Dataset Tab
 * Data paths, augmentation, bucketing, captions
 */
export function DatasetTab({ form, datasets = [], onSave }: TabProps) {
  return (
    <div className="space-y-6">
      <DatasetCard form={form} datasets={datasets} onSave={onSave} />
      <CaptionCard form={form} onSave={onSave} />
      <AugmentationCard form={form} onSave={onSave} />
    </div>
  );
}

/**
 * LoRA Tab
 * Network architecture and structure
 */
export function LoRATab({ form, onSave }: TabProps) {
  return (
    <div className="space-y-6">
      <LoRAStructureCard form={form} onSave={onSave} />
    </div>
  );
}

/**
 * Learning Tab
 * Learning rates and optimizer
 */
export function LearningTab({ form, onSave }: TabProps) {
  return (
    <div className="space-y-6">
      <LearningRateCard form={form} onSave={onSave} />
      <OptimizerCard form={form} onSave={onSave} />
    </div>
  );
}

/**
 * Performance Tab
 * Memory optimization and performance settings
 */
export function PerformanceTab({ form, onSave }: TabProps) {
  return (
    <div className="space-y-6">
      <MemoryCard form={form} onSave={onSave} />
    </div>
  );
}

/**
 * Advanced Tab
 * Advanced training techniques (noise, SNR, loss functions)
 */
export function AdvancedTab({ form, onSave }: TabProps) {
  return (
    <div className="space-y-6">
      <AdvancedCard form={form} onSave={onSave} />
    </div>
  );
}

/**
 * Saving Tab
 * Checkpoint saving and sample generation
 */
export function SavingTab({ form, onSave }: TabProps) {
  return (
    <div className="space-y-6">
      <SavingCard form={form} onSave={onSave} />
    </div>
  );
}
