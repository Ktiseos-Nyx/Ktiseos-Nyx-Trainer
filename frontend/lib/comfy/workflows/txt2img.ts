/**
 * txt2img workflow builder.
 *
 * Produces a ComfyUI workflow prompt for text-to-image generation.
 * Supports SDXL, SD 1.5, and Flux checkpoints via the standard
 * CheckpointLoaderSimple → CLIP → KSampler → VAEDecode → SaveImage chain.
 *
 * LoRA stacks are injected between the checkpoint and the CLIP text encoder
 * using the standard LoraLoader pattern.
 *
 * Node ID scheme (stable, not user-visible):
 *   1  CheckpointLoaderSimple
 *   2  CLIPTextEncode (positive)
 *   3  CLIPTextEncode (negative)
 *   4  EmptyLatentImage
 *   5  KSampler
 *   6  VAEDecode
 *   7  SaveImage
 *   10+ LoraLoader chain (one per LoRA)
 */

import type { ComfyWorkflow } from '../types';

export interface LoraEntry {
  /** Filename as it appears in ComfyUI's loras folder. */
  name: string;
  /** Model weight (strength_model). Default 1.0. */
  modelWeight?: number;
  /** CLIP weight (strength_clip). Default 1.0. */
  clipWeight?: number;
}

export interface Txt2ImgParams {
  /** Checkpoint filename (e.g. "sdxlBase.safetensors"). */
  checkpoint: string;
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
  /**
   * Sampler name. Must match a ComfyUI KSampler sampler_name value.
   * Default "euler_ancestral".
   */
  sampler?: string;
  /**
   * Scheduler name. Must match a ComfyUI KSampler scheduler value.
   * Default "karras".
   */
  scheduler?: string;
  /** Seed. -1 or undefined = randomised by ComfyUI. Default -1. */
  seed?: number;
  /** Batch size (number of images). Default 1. */
  batchSize?: number;
  /** LoRA stack to inject, in order. Applied from first to last. */
  loras?: LoraEntry[];
  /**
   * Denoise strength. 1.0 = full generation (correct for txt2img).
   * Lower values retain more of the input latent — useful when chaining
   * a HiRes-fix or detailer pass. Default 1.0.
   */
  denoise?: number;
  /**
   * Optional external VAE filename. When omitted, the checkpoint's
   * built-in VAE is used.
   */
  vae?: string;
  /**
   * CLIP skip: how many layers from the end of the CLIP text encoder to stop at.
   * Negative integer — -1 = last layer (default), -2 = penultimate layer
   * (common for anime/NAI-style models).
   */
  clipSkip?: number;
  /** Output filename prefix. ComfyUI appends a counter and extension.
   * Default "ComfyUI".
   */
  outputPrefix?: string;
}

/**
 * Build a ComfyUI workflow for txt2img.
 *
 * Returns a workflow object ready to pass to `comfyClient.submitPrompt()`.
 */
export function buildTxt2ImgWorkflow(params: Txt2ImgParams): ComfyWorkflow {
  const {
    checkpoint,
    positivePrompt,
    negativePrompt = '',
    width = 1024,
    height = 1024,
    steps = 20,
    cfg = 7,
    sampler = 'euler_ancestral',
    scheduler = 'karras',
    seed = -1,
    batchSize = 1,
    loras = [],
    denoise = 1.0,
    vae,
    clipSkip = -1,
    outputPrefix = 'ComfyUI',
  } = params;

  const workflow: ComfyWorkflow = {};

  // Node 1: Load checkpoint
  workflow['1'] = {
    class_type: 'CheckpointLoaderSimple',
    inputs: { ckpt_name: checkpoint },
  };

  // Node 8 (optional): External VAE loader
  // When vae is provided, we load it separately and swap it in at decode time.
  const vaeRef: [string, number] = vae ? ['8', 0] : ['1', 2];
  if (vae) {
    workflow['8'] = {
      class_type: 'VAELoader',
      inputs: { vae_name: vae },
    };
  }

  // Node 9 (optional): CLIP skip via CLIPSetLastLayer
  // -1 is the ComfyUI default (no-op), so we only inject the node when the
  // user explicitly requests a different value.
  let clipRef: [string, number] = ['1', 1];
  if (clipSkip !== -1) {
    workflow['9'] = {
      class_type: 'CLIPSetLastLayer',
      inputs: { clip: ['1', 1], stop_at_clip_layer: clipSkip },
    };
    clipRef = ['9', 0];
  }

  // LoRA chain: nodes 10, 11, 12, …
  // Each LoraLoader takes model + clip from the previous node (or the checkpoint).
  // After the chain, modelRef and clipRef point to the last LoraLoader (or checkpoint).
  let modelRef: [string, number] = ['1', 0]; // [nodeId, outputIndex]

  loras.forEach((lora, i) => {
    const nodeId = String(10 + i);
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

  // Node 2: Positive CLIP encode
  workflow['2'] = {
    class_type: 'CLIPTextEncode',
    inputs: { text: positivePrompt, clip: clipRef },
  };

  // Node 3: Negative CLIP encode
  workflow['3'] = {
    class_type: 'CLIPTextEncode',
    inputs: { text: negativePrompt, clip: clipRef },
  };

  // Node 4: Empty latent image
  workflow['4'] = {
    class_type: 'EmptyLatentImage',
    inputs: { width, height, batch_size: batchSize },
  };

  // Node 5: KSampler
  workflow['5'] = {
    class_type: 'KSampler',
    inputs: {
      model: modelRef,
      positive: ['2', 0],
      negative: ['3', 0],
      latent_image: ['4', 0],
      seed,
      steps,
      cfg,
      sampler_name: sampler,
      scheduler,
      denoise,
    },
  };

  // Node 6: VAE decode — uses external VAE if provided, else checkpoint's built-in
  workflow['6'] = {
    class_type: 'VAEDecode',
    inputs: {
      samples: ['5', 0],
      vae: vaeRef,
    },
  };

  // Node 7: Save image
  workflow['7'] = {
    class_type: 'SaveImage',
    inputs: {
      images: ['6', 0],
      filename_prefix: outputPrefix,
    },
  };

  return workflow;
}
