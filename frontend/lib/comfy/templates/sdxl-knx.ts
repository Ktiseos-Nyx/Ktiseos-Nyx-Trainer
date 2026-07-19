/**
 * Node map and patch builder for the Ecosystem SDXL template.
 *
 * Template: sdxl-knx-v13pt5.json (4 Adetailer chains: face 77, eye 130, hand 146, mouth 152)
 * Author:   Ecosystem (forked from Guy90s's ANIMA workflow)
 * Model:    SDXL / NoobAI-XL (Checkpoint Loader LoraManager)
 *
 * Key difference from the ANIMA template:
 *   - Uses `Checkpoint Loader (LoraManager)` (comfyui_fearnworksnodes) which
 *     loads MODEL + CLIP + VAE from a single checkpoint file — no UNETLoader.
 *   - Uses a SEPARATE VAELoader for VAEDecode only (the illustriousXL VAE).
 *     Adetailer and Upscale use the checkpoint's built-in VAE.
 *   - ModelSamplingAuraFlow and the ANIMA-specific loaders are BYPASSED
 *     (kept in the JSON as history, not executed).
 *   - Upscale (UltimateSDUpscale) and Adetailer (DetailerForEach) are ACTIVE
 *     in this template — they always run.
 *
 * Required custom nodes (COMFY-7):
 *   - comfyui_fearnworksnodes   (Checkpoint Loader, Prompt node)
 *   - comfyui-lora-manager      (Lora Loader + Save Image)
 *   - rgthree-comfy             (Seed node)
 *   - comfyui-impact-pack       (DetailerForEach, ImpactSimpleDetectorSEGS, SEGSPreview)
 *   - comfyui-impact-subpack    (UltralyticsDetectorProvider)
 *   - ComfyUI_UltimateSDUpscale (UltimateSDUpscale)
 *
 * Node IDs (from the JSON — do not renumber):
 *   118  Checkpoint Loader (LoraManager) → MODEL[0], CLIP[1], VAE[2]
 *   17   VAELoader (illustrious XL VAE)  → VAE (for VAEDecode only)
 *   64   Lora Loader (LoraManager)       → MODEL, CLIP
 *   98   Seed (rgthree)                  → SEED INT
 *   51   EmptyLatentImage                → LATENT
 *   6    CLIPTextEncode (positive)       → CONDITIONING
 *   7    CLIPTextEncode (negative)       → CONDITIONING
 *   3    KSampler                        → LATENT
 *   8    VAEDecode                       → IMAGE
 *   37   UpscaleModelLoader              → UPSCALE_MODEL
 *   53   UltimateSDUpscale               → IMAGE
 *   77   UltralyticsDetectorProvider     → BBOX_DETECTOR, SEGM_DETECTOR
 *   78   SAMLoader                       → SAM_MODEL
 *   76   ImpactSimpleDetectorSEGS        → SEGS
 *   74   DetailerForEach                 → IMAGE
 *   65   Save Image (LoraManager)
 *
 * Widget index reference (0-based):
 *   118 (Checkpoint Loader LM):  [0]=ckpt_name
 *   17  (VAELoader):             [0]=vae_name
 *   98  (Seed rgthree):          [0]=seed
 *   51  (EmptyLatentImage):      [0]=width, [1]=height, [2]=batch_size
 *   6   (CLIPTextEncode pos):    [0]=text
 *   7   (CLIPTextEncode neg):    [0]=text
 *   3   (KSampler):              [0]=seed*, [1]=ctrl_after_gen (hidden),
 *                                [2]=steps, [3]=cfg, [4]=sampler_name,
 *                                [5]=scheduler, [6]=denoise
 *   37  (UpscaleModelLoader):    [0]=model_name
 *   53  (UltimateSDUpscale):     [0]=upscale_by, [1]=seed*, [2]=ctrl (hidden),
 *                                [3]=steps, [4]=cfg, [5]=sampler_name,
 *                                [6]=scheduler, [7]=denoise, …
 *   74  (DetailerForEach):       many — seed at [3], steps at [5], cfg at [6],
 *                                sampler at [7], scheduler at [8], denoise at [9]
 *   77  (UltralyticsDetectorProvider): [0]=model_name
 *   78  (SAMLoader):             [0]=model_name, [1]=device_mode
 *   65  (Save Image LM):         [0]=filename_prefix, [1]=file_format, …
 */

import type { WorkflowPatch } from '../templateInjector';
import type { LoraEntry } from '../workflows';

export interface SdxlKnxBuildResult {
  patch: WorkflowPatch;
  /**
   * Node IDs to bypass, in topological order (upstream first).
   * Pass to injectTemplate as options.bypassNodeIds.
   */
  bypassNodeIds: number[];
}

export interface SdxlKnxTemplateParams {
  /** SDXL checkpoint filename (e.g. "NoobAI/anime/retirementMixNAIXL_v10.safetensors"). */
  checkpointName?: string;
  /** VAE filename used for primary VAEDecode (e.g. "illustriousXLV10_v10.safetensors"). */
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
  /** KSampler sampler name. Default "euler_ancestral". */
  sampler?: string;
  /** KSampler scheduler. Default "karras". */
  scheduler?: string;
  /** Seed for Seed(rgthree). -1 = random. Default -1. */
  seed?: number;
  /** Batch size. Default 1. */
  batchSize?: number;
  /**
   * LoRAs to apply. Each entry is loaded via the LoRA Manager node's
   * structured `loras` widget — the Python backend ignores the `text` field.
   */
  loras?: LoraEntry[];
  /** Upscale model filename (e.g. "4x_CountryRoads_377000_G.pth"). */
  upscaleModel?: string;
  /** Upscale factor (e.g. 1.5). Default 1.5. */
  upscaleBy?: number;
  /** Face Adetailer detection model — node 77 (e.g. "bbox/face_yolov8m.pt"). */
  adetailerModel?: string;
  /** Eye Adetailer detection model — node 130 (e.g. "segm/Anzhc Eyes -seg-hd.pt"). */
  eyeDetailerModel?: string;
  /** Hand Adetailer detection model — node 146 (e.g. "segm/PitHandDetailer-v1b-seg.pt"). */
  handDetailerModel?: string;
  /** Mouth Adetailer detection model — node 152 (e.g. "bbox/adetailer2dMouth_v10.pt"). */
  mouthDetailerModel?: string;
  /** SAM model for Adetailer segmentation (e.g. "sam_vit_b_01ec64.pth"). */
  samModel?: string;
  /** Output filename prefix. Default "ComfyUI". */
  outputPrefix?: string;
  /** Enable UltimateSDUpscale post-processing. Default true. */
  upscaleEnabled?: boolean;
  /** Enable Adetailer (DetailerForEach + detection nodes). Default true. */
  adetailerEnabled?: boolean;
}

/**
 * Build a patch map for the Ecosystem SDXL template from user parameters.
 *
 * Only supplied parameters are patched. Because Upscale and Adetailer are
 * always active in this template, sampler/cfg/steps patches are applied to
 * all three (KSampler, UltimateSDUpscale, DetailerForEach) so results remain
 * consistent — matching the default behaviour of the original workflow.
 */
export function buildSdxlKnxPatch(params: SdxlKnxTemplateParams): SdxlKnxBuildResult {
  const patch: WorkflowPatch = {};

  // Node 118: Checkpoint Loader (LoraManager)
  if (params.checkpointName !== undefined) {
    patch['118'] = { ...patch['118'], 0: params.checkpointName };
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

  // Node 3: KSampler — [0]=seed, [1]=ctrl_hidden, [2]=steps, [3]=cfg, [4]=sampler, [5]=scheduler, [6]=denoise
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

  // Node 37: UpscaleModelLoader
  if (params.upscaleModel !== undefined) {
    patch['37'] = { ...patch['37'], 0: params.upscaleModel };
  }

  // Node 53: UltimateSDUpscale — [0]=upscale_by; sampler/cfg/steps mirror KSampler
  if (params.upscaleBy !== undefined) {
    patch['53'] = { ...patch['53'], 0: params.upscaleBy };
  }
  if (params.steps !== undefined) {
    patch['53'] = { ...patch['53'], 3: params.steps };
  }
  if (params.cfg !== undefined) {
    patch['53'] = { ...patch['53'], 4: params.cfg };
  }
  if (params.sampler !== undefined) {
    patch['53'] = { ...patch['53'], 5: params.sampler };
  }
  if (params.scheduler !== undefined) {
    patch['53'] = { ...patch['53'], 6: params.scheduler };
  }

  // Nodes 77/130/146/152: UltralyticsDetectorProvider — face/eye/hand/mouth detectors.
  // Each is its own model_name[0] widget; leaving a param undefined keeps the template default.
  if (params.adetailerModel !== undefined) {
    patch['77'] = { ...patch['77'], 0: params.adetailerModel };
  }
  if (params.eyeDetailerModel !== undefined) {
    patch['130'] = { ...patch['130'], 0: params.eyeDetailerModel };
  }
  if (params.handDetailerModel !== undefined) {
    patch['146'] = { ...patch['146'], 0: params.handDetailerModel };
  }
  if (params.mouthDetailerModel !== undefined) {
    patch['152'] = { ...patch['152'], 0: params.mouthDetailerModel };
  }

  // Node 78: SAMLoader — [0]=model_name, [1]=device_mode (leave device_mode at default AUTO)
  if (params.samModel !== undefined) {
    patch['78'] = { ...patch['78'], 0: params.samModel };
  }

  // Node 74: DetailerForEach — only mirror steps/cfg; sampler+scheduler are fixed
  // to euler/karras in the template JSON and must not be overridden by the main
  // KSampler selection (Adetailer works better with a stable sampler pair).
  // Widget order: guide_size[0], guide_size_for[1], max_size[2], seed[3], ctrl[4],
  //               steps[5], cfg[6], sampler_name[7], scheduler[8], denoise[9], …
  if (params.steps !== undefined) {
    patch['74'] = { ...patch['74'], 5: params.steps };
  }
  if (params.cfg !== undefined) {
    patch['74'] = { ...patch['74'], 6: params.cfg };
  }

  // Node 64: Lora Loader (LoraManager)
  // The Python backend reads the 'loras' widget (index 2) as structured objects.
  // The 'text' field (index 1) is an AUTOCOMPLETE UI widget — explicitly deleted
  // by the Python node and NOT used to determine which LoRAs to load.
  // Format per _collect_widget_entries: { name, active, strength, clipStrength }.
  //
  // LM cache indexes files by filename WITH extension. If the user typed a bare
  // stem (e.g. "MyLora") the cache lookup fails and LM falls back to setting
  // absolute_path = "<lora:name:strength>", which then throws FileNotFoundError.
  // Appending ".safetensors" when no recognised extension is present fixes this.
  const LORA_EXTENSIONS = ['.safetensors', '.pt', '.ckpt', '.bin', '.pth'];
  const normalizeLora = (n: string) => {
    // Parse A1111 syntax: <lora:stem:model_weight> or <lora:stem:model_weight:clip_weight>
    const a1111 = n.match(/^<lora:([^:>]+):/i);
    if (a1111) return `${a1111[1]}.safetensors`;
    return LORA_EXTENSIONS.some(ext => n.toLowerCase().endsWith(ext)) ? n : `${n}.safetensors`;
  };

  patch['64'] = {
    ...patch['64'],
    2: (params.loras ?? [])
      .filter(l => l.name.trim())
      .map(l => ({
        name: normalizeLora(l.name.trim()),
        active: true,
        strength: l.modelWeight ?? 1.0,
        clipStrength: l.clipWeight ?? l.modelWeight ?? 1.0,
      })),
  };

  // Node 65: Save Image (LoraManager)
  if (params.outputPrefix !== undefined) {
    patch['65'] = { ...patch['65'], 0: params.outputPrefix };
  }

  // Bypass lists — topological order (upstream nodes first) so rewiring chains correctly.
  // Upscale:   53 (UltimateSDUpscale) before 37 (UpscaleModelLoader, no IMAGE input)
  // Adetailer: 74 (DetailerForEach) first so its downstream is rewired before
  //            76/77/78 are removed.
  const bypassNodeIds: number[] = [];
  if (params.upscaleEnabled === false) {
    bypassNodeIds.push(53, 37);
  }
  if (params.adetailerEnabled === false) {
    // v13pt5 has 4 detailer chains (face/eye/hand/mouth). DetailerForEach nodes in
    // topological IMAGE order (74→132→149→151) so each bypass rewires onto the
    // already-rewired upstream; then their SEGS detectors, detector providers, and
    // the shared SAMLoader (no IMAGE input — simply removed).
    bypassNodeIds.push(74, 132, 149, 151, 76, 131, 143, 153, 77, 130, 146, 152, 78);
  }

  return { patch, bypassNodeIds };
}
