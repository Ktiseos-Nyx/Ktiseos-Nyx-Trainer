'use client';

/**
 * useComfyModels — fetches available model file lists from ComfyUI.
 *
 * Queries ComfyUI's /models/{folder} endpoint for each model type our
 * workflows need. Intended for use inside components that are only rendered
 * when ComfyUI is connected (e.g. the GenerateUI component).
 *
 * Individual folder failures return empty arrays so the rest still loads
 * (e.g. a fresh ComfyUI install won't have every folder).
 *
 * CLIP folder handling: merges `text_encoders/` and `clip/` since different
 * ComfyUI versions use different names for the same folder.
 */

import { useCallback, useEffect, useState } from 'react';
import { comfyClient } from './client';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ComfyModelLists {
  /** True while fetching model lists from ComfyUI. */
  loading: boolean;
  /** Models in diffusion_models/ — ANIMA UNET, Flux, SDXL-VAE-separate, etc. */
  diffusionModels: string[];
  /** Models in checkpoints/ — SDXL / SD 1.5 all-in-one files. */
  checkpoints: string[];
  /**
   * CLIP / text encoder models.
   * Merged from both `text_encoders/` (new ComfyUI) and `clip/` (legacy),
   * deduplicated and sorted.
   */
  clipModels: string[];
  /** Models in vae/ */
  vaeModels: string[];
  /** Models in loras/ — from ComfyUI's /models/loras (only sees the loras/ folder). */
  loraModels: string[];
  /**
   * LoRAs from LoRA Manager's cache — includes every path in extra_model_paths.yaml
   * (e.g. training output/). Empty if LM is not installed. Falls back to `loraModels`
   * in consumers when empty.
   */
  lmLoraModels: string[];
  /** Models in upscale_models/ */
  upscaleModels: string[];
  /** Re-fetch all model lists from ComfyUI. */
  refresh: () => void;
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

/**
 * Fetch available model file lists from ComfyUI's /models/{folder} endpoint.
 * Fetches once on mount; call `refresh()` to re-fetch (e.g. after a download).
 */
export function useComfyModels(): ComfyModelLists {
  const [loading, setLoading] = useState(false);
  const [diffusionModels, setDiffusionModels] = useState<string[]>([]);
  const [checkpoints, setCheckpoints] = useState<string[]>([]);
  const [clipModels, setClipModels] = useState<string[]>([]);
  const [vaeModels, setVaeModels] = useState<string[]>([]);
  const [loraModels, setLoraModels] = useState<string[]>([]);
  const [lmLoraModels, setLmLoraModels] = useState<string[]>([]);
  const [upscaleModels, setUpscaleModels] = useState<string[]>([]);

  const fetchAll = useCallback(async () => {
    /** Fetch a single folder, returning [] on any error. */
    const safe = async (folder: string): Promise<string[]> => {
      try { return await comfyClient.getModelFiles(folder); }
      catch { return []; }
    };

    setLoading(true);
    try {
      const [unet, ckpts, textEnc, clip, vae, loras, upscale, lmLoras] = await Promise.all([
        safe('diffusion_models'),
        safe('checkpoints'),
        safe('text_encoders'),
        safe('clip'),
        safe('vae'),
        safe('loras'),
        safe('upscale_models'),
        comfyClient.getLmLoras(),
      ]);
      setDiffusionModels(unet);
      setCheckpoints(ckpts);
      // Merge text_encoders + clip; different ComfyUI versions use different names.
      setClipModels([...new Set([...textEnc, ...clip])].sort());
      setVaeModels(vae);
      setLoraModels(loras);
      setLmLoraModels(lmLoras);
      setUpscaleModels(upscale);
    } finally {
      setLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // fetchAll is intentionally stable — comfyClient is a module singleton

  useEffect(() => { void fetchAll(); }, [fetchAll]);

  return {
    loading,
    diffusionModels,
    checkpoints,
    clipModels,
    vaeModels,
    loraModels,
    lmLoraModels,
    upscaleModels,
    refresh: fetchAll,
  };
}
