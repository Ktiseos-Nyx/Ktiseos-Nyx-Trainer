/**
 * ANIMA workflow builder (programmatic fallback).
 *
 * Produces a minimal ComfyUI API prompt for ANIMA / AuraFlow checkpoints when
 * the full bundled template (anima-guy90s-v10.json) cannot be used because
 * required custom nodes are missing.
 *
 * Key differences from the SDXL / SD 1.5 chain:
 *   - UNETLoader instead of CheckpointLoaderSimple (MODEL only; no CLIP/VAE)
 *   - Single CLIPLoader (type "cosmos" for ANIMA v10, Qwen text encoder)
 *   - Separate VAELoader
 *   - ModelSamplingAuraFlow patches the sigma schedule (shift=5) before KSampler
 *
 * Reference: guy90sVerySimpleAndEasyTo_v10.json (tested 2026-05-19)
 *
 * Node ID scheme (stable, not user-visible):
 *   1   UNETLoader
 *   2   CLIPLoader            (single encoder — "cosmos" type for ANIMA v10)
 *   3   VAELoader
 *   4   ModelSamplingAuraFlow (shift=5; patches Node 1 model)
 *   5   CLIPTextEncode        (positive — uses Node 2 CLIP)
 *   6   CLIPTextEncode        (negative)
 *   7   EmptyLatentImage
 *   8   KSampler              (uses patched MODEL from Node 4)
 *   9   VAEDecode
 *   10  SaveImage
 *   20+ LoraLoader chain      (one per LoRA; injected between Node 1 and Node 4)
 */

import type { ComfyWorkflow } from '../types';
import type { LoraEntry } from './txt2img';

export interface AnimaParams {
  /** UNET model filename (e.g. "anima_baseV10.safetensors"). */
  unetName: string;
  /**
   * Text-encoder filename for the CLIPLoader.
   * ANIMA v10 uses a Qwen encoder (e.g. "qwen_3_06b_base.safetensors").
   * Must be in ComfyUI's `models/text_encoders/` or `models/clip/` folder.
   */
  clipName: string;
  /**
   * CLIPLoader type string. "cosmos" is correct for ANIMA v10's Qwen encoder.
   * Default: "cosmos".
   */
  clipType?: string;
  /** VAE filename (e.g. "qwen_image_vae.safetensors"). */
  vaeName: string;
  /** Positive prompt text. */
  positivePrompt: string;
  /** Negative prompt text. */
  negativePrompt?: string;
  /** Image width in pixels. Default 832. */
  width?: number;
  /** Image height in pixels. Default 1216. */
  height?: number;
  /** Number of steps. Default 30. */
  steps?: number;
  /** CFG scale. Default 4. */
  cfg?: number;
  /** Sampler name. Default "dpmpp_2m_sde_gpu". */
  sampler?: string;
  /** Scheduler name. Default "sgm_uniform". */
  scheduler?: string;
  /** Seed. -1 = randomised by ComfyUI. Default -1. */
  seed?: number;
  /** Batch size. Default 1. */
  batchSize?: number;
  /** Denoise strength. 1.0 = full generation. Default 1.0. */
  denoise?: number;
  /**
   * ModelSamplingAuraFlow sigma shift.
   * Value of 5 matches Guy90s's reference workflow for ANIMA v10.
   * Default: 5.
   */
  auraShift?: number;
  /** LoRA stack. Applied from first to last. */
  loras?: LoraEntry[];
  /** Output filename prefix. Default "ComfyUI". */
  outputPrefix?: string;
}

/**
 * Build a minimal ComfyUI API prompt for ANIMA / AuraFlow checkpoints.
 *
 * This is the programmatic fallback. The primary path uses the bundled
 * guy90s workflow template with the full custom node stack.
 */
export function buildAnimaWorkflow(params: AnimaParams): ComfyWorkflow {
  const {
    unetName,
    clipName,
    clipType = 'cosmos',
    vaeName,
    positivePrompt,
    negativePrompt = '',
    width = 832,
    height = 1216,
    steps = 30,
    cfg = 4,
    sampler = 'dpmpp_2m_sde_gpu',
    scheduler = 'sgm_uniform',
    seed = -1,
    batchSize = 1,
    denoise = 1.0,
    auraShift = 5,
    loras = [],
    outputPrefix = 'ComfyUI',
  } = params;

  const workflow: ComfyWorkflow = {};

  // Node 1: Load UNET model only
  workflow['1'] = {
    class_type: 'UNETLoader',
    inputs: { unet_name: unetName, weight_dtype: 'default' },
  };

  // Node 2: Single CLIP text encoder (Qwen/cosmos for ANIMA v10)
  workflow['2'] = {
    class_type: 'CLIPLoader',
    inputs: { clip_name: clipName, type: clipType },
  };

  // Node 3: VAE
  workflow['3'] = {
    class_type: 'VAELoader',
    inputs: { vae_name: vaeName },
  };

  // LoRA chain: nodes 20, 21, 22, …
  // Injected between UNETLoader and ModelSamplingAuraFlow.
  let modelRef: [string, number] = ['1', 0];
  let clipRef: [string, number] = ['2', 0];

  loras.forEach((lora, i) => {
    const nodeId = String(20 + i);
    workflow[nodeId] = {
      class_type: 'LoraLoader',
      inputs: {
        model: modelRef,
        clip: clipRef,
        lora_name: lora.name,
        strength_model: lora.modelWeight ?? 1.0,
        strength_clip: lora.clipWeight ?? 1.0,
      },
    };
    modelRef = [nodeId, 0];
    clipRef = [nodeId, 1];
  });

  // Node 4: Patch model sigma schedule for AuraFlow's non-standard distribution
  workflow['4'] = {
    class_type: 'ModelSamplingAuraFlow',
    inputs: { model: modelRef, shift: auraShift },
  };

  // Node 5 / 6: CLIP text encode
  workflow['5'] = {
    class_type: 'CLIPTextEncode',
    inputs: { text: positivePrompt, clip: clipRef },
  };
  workflow['6'] = {
    class_type: 'CLIPTextEncode',
    inputs: { text: negativePrompt, clip: clipRef },
  };

  // Node 7: Empty latent image
  workflow['7'] = {
    class_type: 'EmptyLatentImage',
    inputs: { width, height, batch_size: batchSize },
  };

  // Node 8: KSampler — uses patched model from ModelSamplingAuraFlow
  workflow['8'] = {
    class_type: 'KSampler',
    inputs: {
      model: ['4', 0],
      positive: ['5', 0],
      negative: ['6', 0],
      latent_image: ['7', 0],
      seed,
      steps,
      cfg,
      sampler_name: sampler,
      scheduler,
      denoise,
    },
  };

  // Node 9: VAE decode
  workflow['9'] = {
    class_type: 'VAEDecode',
    inputs: { samples: ['8', 0], vae: ['3', 0] },
  };

  // Node 10: Save image
  workflow['10'] = {
    class_type: 'SaveImage',
    inputs: { images: ['9', 0], filename_prefix: outputPrefix },
  };

  return workflow;
}
