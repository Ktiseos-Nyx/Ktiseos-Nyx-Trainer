/**
 * img2img workflow builder.
 *
 * Produces a ComfyUI workflow for image-to-image generation.
 * Loads an input image from ComfyUI's input folder, encodes it into latent
 * space via VAEEncode, then samples with a given denoise strength.
 *
 * Node ID scheme:
 *   1  CheckpointLoaderSimple
 *   2  CLIPTextEncode (positive)
 *   3  CLIPTextEncode (negative)
 *   4  LoadImage
 *   5  VAEEncode
 *   6  KSampler
 *   7  VAEDecode
 *   8  SaveImage
 *   10+ LoraLoader chain (one per LoRA)
 */

import type { ComfyWorkflow } from '../types';
import type { LoraEntry } from './txt2img';

export interface Img2ImgParams {
  /** Checkpoint filename. */
  checkpoint: string;
  /**
   * Input image filename as known to ComfyUI's LoadImage node.
   * Must have been uploaded to ComfyUI's input folder beforehand
   * via POST /upload/image.
   */
  inputImage: string;
  /** Positive prompt text. */
  positivePrompt: string;
  /** Negative prompt text. */
  negativePrompt?: string;
  /**
   * Denoise strength: 0 = identical to input, 1 = full generation.
   * Typical range 0.4–0.8 for img2img. Default 0.75.
   */
  denoise?: number;
  /** Number of steps. Default 20. */
  steps?: number;
  /** CFG scale. Default 7. */
  cfg?: number;
  /** Sampler name. Default "euler_ancestral". */
  sampler?: string;
  /** Scheduler name. Default "karras". */
  scheduler?: string;
  /** Seed. -1 = randomised. Default -1. */
  seed?: number;
  /** LoRA stack to inject. */
  loras?: LoraEntry[];
  /** Output filename prefix. Default "ComfyUI_img2img". */
  outputPrefix?: string;
}

/**
 * Build a ComfyUI workflow for img2img.
 *
 * Returns a workflow object ready to pass to `comfyClient.submitPrompt()`.
 */
export function buildImg2ImgWorkflow(params: Img2ImgParams): ComfyWorkflow {
  const {
    checkpoint,
    inputImage,
    positivePrompt,
    negativePrompt = '',
    denoise = 0.75,
    steps = 20,
    cfg = 7,
    sampler = 'euler_ancestral',
    scheduler = 'karras',
    seed = -1,
    loras = [],
    outputPrefix = 'ComfyUI_img2img',
  } = params;

  const workflow: ComfyWorkflow = {};

  // Node 1: Load checkpoint
  workflow['1'] = {
    class_type: 'CheckpointLoaderSimple',
    inputs: { ckpt_name: checkpoint },
  };

  // LoRA chain: nodes 10+
  let modelRef: [string, number] = ['1', 0];
  let clipRef: [string, number] = ['1', 1];

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

  // Node 4: Load input image
  workflow['4'] = {
    class_type: 'LoadImage',
    inputs: { image: inputImage },
  };

  // Node 5: VAE encode (image → latent)
  workflow['5'] = {
    class_type: 'VAEEncode',
    inputs: {
      pixels: ['4', 0],
      vae: ['1', 2],
    },
  };

  // Node 6: KSampler
  workflow['6'] = {
    class_type: 'KSampler',
    inputs: {
      model: modelRef,
      positive: ['2', 0],
      negative: ['3', 0],
      latent_image: ['5', 0],
      seed,
      steps,
      cfg,
      sampler_name: sampler,
      scheduler,
      denoise,
    },
  };

  // Node 7: VAE decode
  workflow['7'] = {
    class_type: 'VAEDecode',
    inputs: {
      samples: ['6', 0],
      vae: ['1', 2],
    },
  };

  // Node 8: Save image
  workflow['8'] = {
    class_type: 'SaveImage',
    inputs: {
      images: ['7', 0],
      filename_prefix: outputPrefix,
    },
  };

  return workflow;
}
