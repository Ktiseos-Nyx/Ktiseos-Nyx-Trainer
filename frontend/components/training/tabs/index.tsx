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
}

/**
 * Setup Tab
 * Project information and model selection
 */
export function SetupTab({ form }: TabProps) {
  return (
    <div className="space-y-6">
      <ProjectSetupCard form={form} />
    </div>
  );
}

/**
 * Dataset Tab
 * Data paths, augmentation, bucketing
 */
export function DatasetTab({ form }: TabProps) {
  return (
    <div className="space-y-6">
      <DatasetCard form={form} />
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
 * Advanced Tab
 * Caption settings, memory, advanced techniques
 */
export function AdvancedTab({ form }: TabProps) {
  return (
    <div className="space-y-6">
      <CaptionCard form={form} />
      <MemoryCard form={form} />
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
