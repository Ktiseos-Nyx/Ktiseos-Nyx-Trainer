/**
 * Auto-tagging / captioning form hook — RHF as single source of truth.
 *
 * Mirrors useTrainingForm (React Hook Form + zodResolver). On top of the form it
 * owns the dataset list and the *selection repair* logic that fixes the long-
 * standing "it tagged the wrong folder" bug:
 *   - hidden dot-dirs (notably Jupyter's `.ipynb_checkpoints`) are filtered out of
 *     the list entirely, so they can never be picked or become the default;
 *   - the user's selection is sticky — we only auto-pick a default when the current
 *     selection is empty or no longer exists in the (filtered) list;
 *   - the dataset list is fetched once on mount (not on every selection change),
 *     killing the re-fetch loop the old page had.
 */

'use client';

import { useForm, UseFormReturn } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useCallback, useEffect, useState } from 'react';
import { TaggingConfigSchema } from '@/lib/validation';
import { datasetAPI } from '@/lib/api';
import type { DatasetInfo, TaggingConfig } from '@/lib/api';

/** Default auto-tag config — carried over verbatim from the original page state. */
export const defaultTaggingConfig: TaggingConfig = {
  datasetDir: '',
  model: 'SmilingWolf/wd-eva02-large-tagger-v3',
  captionExtension: '.txt',
  captionSeparator: ', ',

  threshold: 0.35,
  useGeneralThreshold: false,
  generalThreshold: 0.35,
  useCharacterThreshold: false,
  characterThreshold: 0.35,

  undesiredTags: '',
  tagReplacement: '',
  alwaysFirstTags: '',
  characterTagsFirst: false,
  ratingTags: 'none',
  removeUnderscore: true,
  characterTagExpand: false,

  blipBeamSearch: false,
  blipNumBeams: 3,
  blipTopP: 0.9,
  blipMaxLength: 75,
  blipMinLength: 5,

  gitMaxLength: 50,
  gitRemoveWords: true,

  overwriteMode: 'overwrite',
  recursive: false,

  batchSize: 8,
  maxWorkers: 2,
  useOnnx: true,
  forceDownload: false,

  frequencyTags: false,
  debug: false,
};

/** A dataset folder is "real" if its name isn't a hidden dot-dir (e.g. `.ipynb_checkpoints`). */
function isRealDataset(d: DatasetInfo): boolean {
  return !d.name.startsWith('.');
}

export interface UseTaggingFormResult {
  form: UseFormReturn<TaggingConfig>;
  datasets: DatasetInfo[];
  loadingDatasets: boolean;
  reloadDatasets: () => Promise<void>;
}

export function useTaggingForm(): UseTaggingFormResult {
  const form = useForm<TaggingConfig>({
    resolver: zodResolver(TaggingConfigSchema),
    defaultValues: defaultTaggingConfig,
  });

  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState(true);

  // `form` is referentially stable across renders, so listing it here does NOT
  // re-create the callback on selection changes — no re-fetch loop.
  const reloadDatasets = useCallback(async () => {
    setLoadingDatasets(true);
    try {
      const data = await datasetAPI.list();
      const real = (data.datasets || []).filter(isRealDataset);
      setDatasets(real);

      // Repair the selection only when it's empty or points at something that no
      // longer exists. A valid user pick is left untouched (sticky).
      const current = form.getValues('datasetDir');
      const stillValid = !!current && real.some((d) => d.path === current);
      if (!stillValid && real.length > 0) {
        form.setValue('datasetDir', real[0].path, { shouldDirty: false });
      }
    } catch (err) {
      console.error('Failed to load datasets:', err);
    } finally {
      setLoadingDatasets(false);
    }
  }, [form]);

  useEffect(() => {
    reloadDatasets();
  }, [reloadDatasets]);

  return { form, datasets, loadingDatasets, reloadDatasets };
}
