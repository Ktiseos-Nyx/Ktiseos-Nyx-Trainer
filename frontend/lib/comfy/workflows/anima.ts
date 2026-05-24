/**
 * ANIMA workflow builder.
 *
 * Produces a ComfyUI workflow for AuraFlow / ANIMA-style checkpoints.
 *
 * Unlike the SDXL/SD 1.5 chain (CheckpointLoaderSimple → CLIP → KSampler),
 * ANIMA loads MODEL, CLIP, and VAE separately and patches the model's sigma
 * schedule with ModelSamplingAuraFlow before sampling.
 *
 * Reference workflow: `guy90sVerySimpleAndEasyTo_v10.json` (tested 2026-05-19)
 *
 * Node ID scheme:
 *   1   UNETLoader            (MODEL output)
 *   2   DualCLIPLoader        (CLIP output — loads CLIP-L + T5XXL for AuraFlow)
 *   3   VAELoader             (VAE output)
 *   4   ModelSamplingAuraFlow (patches Node 1 model; outputs patched MODEL)
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
  /** UNET model filename (e.g. "aura_flow_0.2.safetensors"). */
  unetName: string;
  /**
   * First CLIP model filename. For AuraFlow, this is typically CLIP-L.
   * Must match a file in ComfyUI's `models/clip/` folder.
   */
  clipName1: string;
  /**
   * Second CLIP model filename. For AuraFlow, this is T5XXL.
   * Must match a file in ComfyUI's `models/clip/` folder.
   */
  clipName2: string;
  /** VAE filename. Must match a file in ComfyUI's `models/vae/` folder. */
  vaeName: string;
  /**
   * CLIP type passed to DualCLIPLoader.
   * "stable_diffusion" works for AuraFlow 0.1/0.2.
   * May need "flux" for newer AuraFlow variants — check ComfyUI docs.
   */
  clipType?: 'stable_diffusion' | 'flux';
  /** Positive prompt text. */
  positivePrompt: string;
  /** Negative prompt text. */
  negativePrompt?: string;
  /** Image width in pixels. Default 1024. */
  width?: number;
  /** Image height in pixels. Default 1024. */
  height?: number;
  /** Number of steps. Default 20. */
  steps?: number;
  /** CFG scale. Default 7. */
  cfg?: number;
  /** Sampler name. Default "euler". */
  sampler?: string;
  /** Scheduler name. Default "simple". */
  scheduler?: string;
  /** Seed. -1 or undefined = randomised by ComfyUI. Default -1. */
  seed?: number;
  /** Batch size (number of images). Default 1. */
  batchSize?: number;
  /** Denoise strength. 1.0 = full generation. Default 1.0. */
  denoise?: number;
  /** LoRA stack injected between UNETLoader and ModelSamplingAuraFlow. */
  loras?: LoraEntry[];
  /** Output filename prefix. Default "ComfyUI". */
  outputPrefix?: string;
}

/**
 * Build a ComfyUI workflow for ANIMA / AuraFlow-style checkpoints.
 *
 * Returns a workflow object ready to pass to `comfyClient.submitPrompt()`.
 */
export function buildAnimaWorkflow(params: AnimaParams): ComfyWorkflow {
  const {
    unetName,
    clipName1,
    clipName2,
    vaeName,
    clipType = 'stable_diffusion',
    positivePrompt,
    negativePrompt = '',
    width = 1024,
    height = 1024,
    steps = 20,
    cfg = 7,
    sampler = 'euler',
    scheduler = 'simple',
    seed = -1,
    batchSize = 1,
    denoise = 1.0,
    loras = [],
    outputPrefix = 'ComfyUI',
  } = params;

  const workflow: ComfyWorkflow = {};

  // Node 1: Load UNET model only
  workflow['1'] = {
    class_type: 'UNETLoader',
    inputs: {
      unet_name: unetName,
      weight_dtype: 'default',
    },
  };

  // Node 2: Load dual CLIP (CLIP-L + T5XXL for AuraFlow)
  workflow['2'] = {
    class_type: 'DualCLIPLoader',
    inputs: {
      clip_name1: clipName1,
      clip_name2: clipName2,
      type: clipType,
    },
  };

  // Node 3: Load VAE
  workflow['3'] = {
    class_type: 'VAELoader',
    inputs: { vae_name: vaeName },
  };

  // LoRA chain: nodes 20, 21, 22, …
  // Injected between UNETLoader and ModelSamplingAuraFlow.
  // Each LoraLoader takes model + clip from the previous node (or checkpoint).
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
    inputs: {
      model: modelRef,
      shift: 1.73, // AuraFlow default shift — matches reference workflow
    },
  };

  // Node 5: Positive CLIP encode
  workflow['5'] = {
    class_type: 'CLIPTextEncode',
    inputs: { text: positivePrompt, clip: clipRef },
  };

  // Node 6: Negative CLIP encode
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
    inputs: {
      samples: ['8', 0],
      vae: ['3', 0],
    },
  };

  // Node 10: Save image
  workflow['10'] = {
    class_type: 'SaveImage',
    inputs: {
      images: ['9', 0],
      filename_prefix: outputPrefix,
    },
  };

  return workflow;
}
