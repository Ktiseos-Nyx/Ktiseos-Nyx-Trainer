/**
 * Training Configuration Tab Components
 * Each tab composes related configuration cards
 */

'use client';

import { UseFormReturn } from 'react-hook-form';
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

interface TabProps {
  form: UseFormReturn<Partial<TrainingConfig>>;
  models?: { value: string; label: string }[];
  vaes?: { value: string; label: string }[];
  datasets?: { value: string; label: string }[];
}

/**
 * Setup Tab
 * Project information and model selection
 */
export function SetupTab({ form, models = [], vaes = [] }: TabProps) {
  return (
    <div className="space-y-6">
      <ProjectSetupCard form={form} models={models} vaes={vaes} />
    </div>
  );
}

/**
 * Dataset Tab
 * Data paths, augmentation, bucketing, captions
 */
export function DatasetTab({ form, datasets = [] }: TabProps) {
  return (
    <div className="space-y-6">
      <DatasetCard form={form} datasets={datasets} />
      <CaptionCard form={form} />
      <AugmentationCard form={form} />
    </div>
  );
}

/**
 * LoRA Tab
 * Network architecture and structure
 */
export function LoRATab({ form }: TabProps) {
  return (
    <div className="space-y-6">
      <LoRAStructureCard form={form} />
    </div>
  );
}

/**
 * Learning Tab
 * Learning rates and optimizer
 */
export function LearningTab({ form }: TabProps) {
  return (
    <div className="space-y-6">
      <LearningRateCard form={form} />
      <OptimizerCard form={form} />
    </div>
  );
}

/**
 * Performance Tab
 * Memory optimization and performance settings
 */
export function PerformanceTab({ form }: TabProps) {
  return (
    <div className="space-y-6">
      <MemoryCard form={form} />
    </div>
  );
}

/**
 * Advanced Tab
 * Advanced training techniques (noise, SNR, loss functions)
 */
export function AdvancedTab({ form }: TabProps) {
  return (
    <div className="space-y-6">
      <AdvancedCard form={form} />
    </div>
  );
}

/**
 * Saving Tab
 * Checkpoint saving and sample generation
 */
export function SavingTab({ form }: TabProps) {
  return (
    <div className="space-y-6">
      <SavingCard form={form} />
    </div>
  );
}
