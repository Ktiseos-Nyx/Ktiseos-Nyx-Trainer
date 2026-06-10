/**
 * Auto-tagging / captioning model catalog.
 *
 * Shared between the auto-tag page and its cards so the model list and the
 * "which model family is selected" logic have a single source of truth.
 */

export type TaggingModelType = 'wd14' | 'blip' | 'git';

export interface TaggingModelOption {
  id: string;
  name: string;
  type: TaggingModelType;
  description?: string;
}

export const AVAILABLE_MODELS: TaggingModelOption[] = [
  // WD14 v3 (newest, best quality)
  { id: 'SmilingWolf/wd-eva02-large-tagger-v3', name: 'WD14 EVA02 Large v3', type: 'wd14', description: '⭐ Recommended' },
  { id: 'SmilingWolf/wd-vit-large-tagger-v3', name: 'WD14 ViT Large v3', type: 'wd14' },
  { id: 'SmilingWolf/wd-swinv2-tagger-v3', name: 'WD14 SwinV2 v3', type: 'wd14' },
  { id: 'SmilingWolf/wd-vit-tagger-v3', name: 'WD14 ViT v3', type: 'wd14' },
  // WD14 v2/v1 (older, stable)
  { id: 'SmilingWolf/wd-v1-4-swinv2-tagger-v2', name: 'WD14 SwinV2 v2', type: 'wd14' },
  { id: 'SmilingWolf/wd-v1-4-convnext-tagger-v2', name: 'WD14 ConvNext v2', type: 'wd14' },
  { id: 'SmilingWolf/wd-v1-4-convnext-tagger', name: 'WD14 ConvNext v1', type: 'wd14' },
  { id: 'SmilingWolf/wd-v1-4-vit-tagger-v2', name: 'WD14 ViT v2', type: 'wd14' },
  { id: 'SmilingWolf/wd-v1-4-vit-tagger', name: 'WD14 ViT v1', type: 'wd14' },
  // BLIP
  { id: 'blip-base', name: 'BLIP Base', type: 'blip', description: 'Natural language captions' },
  // GIT
  { id: 'microsoft/git-large-textcaps', name: 'GIT Large (TextCaps)', type: 'git', description: '⭐ Recommended' },
  { id: 'microsoft/git-large', name: 'GIT Large', type: 'git' },
  { id: 'microsoft/git-base', name: 'GIT Base', type: 'git', description: 'Faster' },
];

/** Resolve a model id to its family (defaults to wd14 for unknown ids). */
export function getModelType(modelId: string): TaggingModelType {
  return AVAILABLE_MODELS.find((m) => m.id === modelId)?.type ?? 'wd14';
}
