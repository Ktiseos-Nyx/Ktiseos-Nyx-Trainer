/**
 * Node map and patch builder for the ANIMA template.
 *
 * Template: anima-guy90s-v10.json
 * Author:   Guy90 (original), KNX (ported to app)
 * Model:    ANIMA v10 (AuraFlow + Qwen text encoder)
 *
 * Required custom nodes (COMFY-7):
 *   - comfyui-lora-manager   (Lora Loader + Save Image)
 *   - rgthree-comfy          (Seed node)
 *   - comfyui-impact-pack    (Adetailer — optional group, bypassed by default)
 *   - comfyui-impact-subpack (Adetailer detector — optional)
 *   - ComfyUI_UltimateSDUpscale (Upscale — optional group, bypassed by default)
 *
 * Node IDs (from the JSON — do not renumber):
 *   52  UNETLoader                 → MODEL
 *   18  CLIPLoader (cosmos/Qwen)   → CLIP
 *   17  VAELoader                  → VAE
 *   64  Lora Loader (LoraManager)  → MODEL, CLIP
 *   39  ModelSamplingAuraFlow      → patched MODEL
 *   98  Seed (rgthree)             → SEED INT
 *   51  EmptyLatentImage           → LATENT
 *   6   CLIPTextEncode (positive)  → CONDITIONING
 *   7   CLIPTextEncode (negative)  → CONDITIONING
 *   3   KSampler                   → LATENT
 *   8   VAEDecode                  → IMAGE
 *   65  Save Image (LoraManager)
 *
 * Widget index reference (0-based position in each node's widgets_values):
 *   52 (UNETLoader):            [0]=unet_name, [1]=weight_dtype
 *   18 (CLIPLoader):            [0]=clip_name, [1]=type, [2]=device
 *   17 (VAELoader):             [0]=vae_name
 *   98 (Seed rgthree):          [0]=seed
 *   51 (EmptyLatentImage):      [0]=width, [1]=height, [2]=batch_size
 *   6  (CLIPTextEncode pos):    [0]=text
 *   7  (CLIPTextEncode neg):    [0]=text
 *   3  (KSampler):              [0]=seed*, [1]=control_after_generate (hidden),
 *                               [2]=steps, [3]=cfg, [4]=sampler_name,
 *                               [5]=scheduler, [6]=denoise
 *                               (* seed is overridden by Seed(rgthree) link)
 *   39 (ModelSamplingAuraFlow): [0]=shift
 *   65 (Save Image LoraManager): [0]=filename_prefix, [1]=file_format, …
 */

import type { WorkflowPatch } from '../templateInjector';
import type { LoraEntry } from '../workflows';

export interface AnimaTemplateParams {
  /** ANIMA UNET filename (e.g. "anima_baseV10.safetensors"). */
  unetName?: string;
  /** Qwen text encoder filename (e.g. "qwen_3_06b_base.safetensors"). */
  clipName?: string;
  /** VAE filename (e.g. "qwen_image_vae.safetensors"). */
  vaeName?: string;
  /** Positive prompt. */
  positivePrompt?: string;
  /** Negative prompt. */
  negativePrompt?: string;
  /** Image width. Default 832. */
  width?: number;
  /** Image height. Default 1216. */
  height?: number;
  /** KSampler steps. Default 30. */
  steps?: number;
  /** KSampler CFG. Default 4. */
  cfg?: number;
  /** KSampler sampler name. Default "dpmpp_2m_sde_gpu". */
  sampler?: string;
  /** KSampler scheduler. Default "sgm_uniform". */
  scheduler?: string;
  /** Seed value for Seed(rgthree). -1 = random. Default -1. */
  seed?: number;
  /** Batch size. Default 1. */
  batchSize?: number;
  /**
   * LoRAs to apply. Each entry is loaded via the LoRA Manager node's
   * structured `loras` widget — the Python backend ignores the `text` field.
   */
  loras?: LoraEntry[];
  /** AuraFlow sigma shift for ModelSamplingAuraFlow. Default 5. */
  auraShift?: number;
  /** Output filename prefix. Default "ComfyUI". */
  outputPrefix?: string;
}

/**
 * Build a patch map for the ANIMA template from user parameters.
 *
 * Only the supplied parameters are patched — unspecified values remain
 * at the template's defaults.
 */
export function buildAnimaPatch(params: AnimaTemplateParams): WorkflowPatch {
  const patch: WorkflowPatch = {};

  // Node 52: UNETLoader
  if (params.unetName !== undefined) {
    patch['52'] = { ...patch['52'], 0: params.unetName };
  }

  // Node 18: CLIPLoader
  if (params.clipName !== undefined) {
    patch['18'] = { ...patch['18'], 0: params.clipName };
  }

  // Node 17: VAELoader
  if (params.vaeName !== undefined) {
    patch['17'] = { ...patch['17'], 0: params.vaeName };
  }

  // Node 98: Seed (rgthree)
  if (params.seed !== undefined) {
    patch['98'] = { ...patch['98'], 0: params.seed };
  }

  // Node 51: EmptyLatentImage
  if (params.width !== undefined) {
    patch['51'] = { ...patch['51'], 0: params.width };
  }
  if (params.height !== undefined) {
    patch['51'] = { ...patch['51'], 1: params.height };
  }
  if (params.batchSize !== undefined) {
    patch['51'] = { ...patch['51'], 2: params.batchSize };
  }

  // Node 6: CLIPTextEncode (positive)
  if (params.positivePrompt !== undefined) {
    patch['6'] = { ...patch['6'], 0: params.positivePrompt };
  }

  // Node 7: CLIPTextEncode (negative)
  if (params.negativePrompt !== undefined) {
    patch['7'] = { ...patch['7'], 0: params.negativePrompt };
  }

  // Node 3: KSampler
  // Index 0=seed (also linked from Seed rgthree), 1=control_after_generate (hidden), 2=steps, 3=cfg, 4=sampler, 5=scheduler, 6=denoise
  if (params.steps !== undefined) {
    patch['3'] = { ...patch['3'], 2: params.steps };
  }
  if (params.cfg !== undefined) {
    patch['3'] = { ...patch['3'], 3: params.cfg };
  }
  if (params.sampler !== undefined) {
    patch['3'] = { ...patch['3'], 4: params.sampler };
  }
  if (params.scheduler !== undefined) {
    patch['3'] = { ...patch['3'], 5: params.scheduler };
  }

  // Node 39: ModelSamplingAuraFlow
  if (params.auraShift !== undefined) {
    patch['39'] = { ...patch['39'], 0: params.auraShift };
  }

  // Node 64: Lora Loader (LoraManager)
  // Same as SDXL: the Python backend uses the 'loras' widget (index 2),
  // not the 'text' field (index 1) which is deleted before processing.
  // LM cache indexes by filename WITH extension — append .safetensors if missing.
  const LORA_EXTENSIONS = ['.safetensors', '.pt', '.ckpt', '.bin', '.pth'];
  const normalizeLora = (n: string) => {
    // Parse A1111 syntax: <lora:stem:model_weight> or <lora:stem:model_weight:clip_weight>
    const a1111 = n.match(/^<lora:([^:>]+):/i);
    if (a1111) return `${a1111[1]}.safetensors`;
    return LORA_EXTENSIONS.some(ext => n.toLowerCase().endsWith(ext)) ? n : `${n}.safetensors`;
  };

  if (params.loras !== undefined) {
    patch['64'] = {
      ...patch['64'],
      2: params.loras
        .filter(l => l.name.trim())
        .map(l => ({
          name: normalizeLora(l.name.trim()),
          active: true,
          strength: l.modelWeight ?? 1.0,
          clipStrength: l.clipWeight ?? l.modelWeight ?? 1.0,
        })),
    };
  }

  // Node 65: Save Image (LoraManager)
  if (params.outputPrefix !== undefined) {
    patch['65'] = { ...patch['65'], 0: params.outputPrefix };
  }

  return patch;
}
