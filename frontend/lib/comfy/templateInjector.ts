/**
 * ComfyUI template injector.
 *
 * Takes a full ComfyUI workflow JSON (the "graph" format exported from the UI),
 * patches the specified node widget values with user-supplied parameters,
 * and converts it to the API prompt format required by POST /comfyui/prompt.
 *
 * The workflow JSON is also returned so it can be embedded in the submission as
 * extra_data.extra_pnginfo.workflow — this is what causes ComfyUI to save
 * the full workflow inside generated PNG metadata.
 *
 * Two bundled templates:
 *   - anima-guy90s-v10   — Guy90s's ANIMA workflow (AuraFlow, Qwen encoder)
 *   - sdxl-knx-v1        — KNX SDXL fork of the above (CheckpointLoader, NoobAI)
 *
 * Both templates require the custom node stack from COMFY-7. If nodes are missing,
 * callers should fall back to the programmatic builders (buildAnimaWorkflow /
 * buildTxt2ImgWorkflow) which use only built-in ComfyUI nodes.
 */

import type { ComfyWorkflow } from './types';

// ─── Workflow graph types (internal use) ────────────────────────────────────

interface WorkflowNodeInput {
  name: string;
  type: string;
  link?: number | null;
  widget?: { name: string };
  shape?: number;
}

interface WorkflowNode {
  id: number;
  type: string;
  mode: number; // 0 = active, 4 = bypassed
  inputs?: WorkflowNodeInput[];
  outputs?: Array<{ name: string; type: string; links?: number[] | null }>;
  widgets_values?: unknown[];
}

interface WorkflowGraph {
  nodes: WorkflowNode[];
  /** [[linkId, fromNodeId, fromSlot, toNodeId, toSlot, type], ...] */
  links: [number, number, number, number, number, string][];
}

// ─── Widget patch map ────────────────────────────────────────────────────────

/**
 * A patch map describes which widget values to overwrite in which nodes.
 *
 * Structure: `{ [nodeId]: { [widgetIndex]: newValue } }`
 *
 * widgetIndex is the 0-based position in the node's `widgets_values` array.
 * Look up node IDs and widget positions in the template workflow JSON.
 */
export type WorkflowPatch = Record<string, Record<number, unknown>>;

// ─── Node types skipped in API prompt conversion ─────────────────────────────

/** Purely visual nodes that have no execution effect. */
const SKIP_NODE_TYPES = new Set([
  'Note',
  'PreviewImage',
  'Image Comparer (rgthree)',
  'Fast Groups Bypasser (rgthree)',
  'SEGSPreview',
]);

// ─── Widget definitions ───────────────────────────────────────────────────────

/**
 * Per-node-type ordered list of widget input definitions.
 *
 * ComfyUI's graph format stores widget values as an ordered array
 * (`widgets_values`) without names. The names are only known from the
 * node-type definition served by `/object_info`. Rather than querying
 * the server at runtime we maintain a static table for node types used
 * in our bundled templates.
 *
 * Fields:
 *   name        — API input name (must match ComfyUI's INPUT_TYPES definition)
 *   hidden      — UI-only value; consume the widgets_values slot but omit from
 *                 the API prompt (e.g. `control_after_generate` in KSampler)
 *   connectable — widget that CAN accept a link (also appears in node.inputs
 *                 with a `widget` property); may be a link ref OR a widget value
 *                 depending on whether it's connected in the graph
 *
 * Only nodes with REQUIRED pure-widget inputs need an entry here. Nodes whose
 * widget inputs are either all connectable (already handled via inputs array) or
 * all optional/defaulted by ComfyUI work without an entry.
 *
 * TODO: Replace with dynamic resolution via GET /object_info to avoid
 *       version-drift issues as custom nodes update.
 */
interface WidgetDef {
  name: string;
  hidden?: boolean;
  connectable?: boolean;
}

const WIDGET_DEFS: Record<string, WidgetDef[]> = {
  // ─── Standard ComfyUI nodes ─────────────────────────────────────────────
  KSampler: [
    { name: 'seed', connectable: true },
    { name: 'control_after_generate', hidden: true },
    { name: 'steps' },
    { name: 'cfg' },
    { name: 'sampler_name' },
    { name: 'scheduler' },
    { name: 'denoise' },
  ],
  KSamplerAdvanced: [
    { name: 'add_noise' },
    { name: 'noise_seed', connectable: true },
    { name: 'control_before_generate', hidden: true },
    { name: 'steps' },
    { name: 'start_at_step' },
    { name: 'end_at_step' },
    { name: 'return_with_leftover_noise' },
    { name: 'cfg' },
    { name: 'sampler_name' },
    { name: 'scheduler' },
  ],
  CheckpointLoaderSimple: [{ name: 'ckpt_name' }],
  VAELoader: [{ name: 'vae_name' }],
  UNETLoader: [{ name: 'unet_name' }, { name: 'weight_dtype' }],
  // CLIPLoader v0.3+ added a `device` parameter for CPU offloading.
  CLIPLoader: [{ name: 'clip_name' }, { name: 'type' }, { name: 'device' }],
  DualCLIPLoader: [{ name: 'clip_name1' }, { name: 'clip_name2' }, { name: 'type' }],
  TripleCLIPLoader: [{ name: 'clip_name1' }, { name: 'clip_name2' }, { name: 'clip_name3' }],
  // ANIMA workflow uses AuraFlow shift parameter.
  ModelSamplingAuraFlow: [{ name: 'shift' }],
  ModelSamplingFlux: [{ name: 'max_shift' }, { name: 'base_shift' }],
  EmptyLatentImage: [{ name: 'width' }, { name: 'height' }, { name: 'batch_size' }],
  EmptySD3LatentImage: [{ name: 'width' }, { name: 'height' }, { name: 'batch_size' }],
  CLIPTextEncode: [{ name: 'text' }],
  LoraLoader: [
    { name: 'lora_name' },
    { name: 'strength_model' },
    { name: 'strength_clip' },
  ],
  UpscaleModelLoader: [{ name: 'model_name' }],
  SaveImage: [{ name: 'filename_prefix' }],

  // ─── rgthree-comfy ───────────────────────────────────────────────────────
  // Seed (rgthree): outputs a single INT seed. The extra entries in
  // widgets_values (labels, display mode) are UI-only — stop after `seed`.
  'Seed (rgthree)': [{ name: 'seed' }],

  // ─── LoRA Manager (comfyui-lora-manager) ────────────────────────────────
  'Checkpoint Loader (LoraManager)': [{ name: 'ckpt_name' }],
  'Lora Loader (LoraManager)': [
    // Index 0 is a UI autocomplete metadata blob — not a real API input.
    { name: '__lm_autocomplete_meta_text', hidden: true },
    { name: 'text' },
    { name: 'loras' },
  ],
  // widgets_values: [filename_prefix, file_format, lossless_webp, quality,
  //   save_workflow, save_civitai_info, save_thumbnail, use_trigger_words]
  // Indices 4-7 are best-guess names; if wrong they are treated as unknown
  // optional inputs by ComfyUI and ignored rather than causing a 400.
  'Save Image (LoraManager)': [
    { name: 'filename_prefix' },
    { name: 'file_format' },
    { name: 'lossless_webp' },
    { name: 'quality' },
    { name: 'save_workflow' },
    { name: 'save_civitai_info' },
    { name: 'save_thumbnail' },
    { name: 'use_trigger_words' },
  ],

  // ─── UltimateSDUpscale (comfyui_ultimatesdupscale) ───────────────────────
  // Parameter list from usdu_nodes.py INPUT_TYPES (v1.7+).
  // upscale_model is a linked input (in node.inputs), not a widget.
  UltimateSDUpscale: [
    { name: 'upscale_by' },
    { name: 'seed', connectable: true },
    { name: 'control_after_generate', hidden: true },
    { name: 'steps' },
    { name: 'cfg' },
    { name: 'sampler_name' },
    { name: 'scheduler' },
    { name: 'denoise' },
    { name: 'mode_type' },
    { name: 'tile_width' },
    { name: 'tile_height' },
    { name: 'mask_blur' },
    { name: 'tile_padding' },
    { name: 'seam_fix_mode' },
    { name: 'seam_fix_denoise' },
    { name: 'seam_fix_width' },
    { name: 'seam_fix_mask_blur' },
    { name: 'seam_fix_padding' },
    { name: 'force_uniform_tiles' },
    { name: 'tiled_decode' },
    { name: 'batch_size' },
  ],

  // ─── Impact Pack (comfyui-impact-pack) ───────────────────────────────────
  UltralyticsDetectorProvider: [{ name: 'model_name' }],
  SAMLoader: [{ name: 'model_name' }, { name: 'device_mode' }],
  // DetailerForEach: widgets_values (19 entries) mapped from node 74 in sdxl-knx-v1.json.
  // Indices 17-18 (both false) are unknown optional params in Impact Pack 8.28.2 —
  // omitted here so ComfyUI uses their defaults.
  DetailerForEach: [
    { name: 'guide_size' },
    { name: 'guide_size_for' },
    { name: 'max_size' },
    { name: 'seed', connectable: true },
    { name: 'control_after_generate', hidden: true },
    { name: 'steps' },
    { name: 'cfg' },
    { name: 'sampler_name' },
    { name: 'scheduler' },
    { name: 'denoise' },
    { name: 'feather' },
    { name: 'noise_mask' },
    { name: 'force_inpaint' },
    { name: 'wildcard' },
    { name: 'cycle' },
    { name: 'inpaint_model' },
    { name: 'noise_mask_feather' },
  ],
};

// ─── Core functions ──────────────────────────────────────────────────────────

/**
 * Apply a patch map to a deep-cloned workflow graph.
 * Only `widgets_values` is mutated — structure and links are untouched.
 */
function applyPatch(graph: WorkflowGraph, patch: WorkflowPatch): WorkflowGraph {
  const cloned: WorkflowGraph = JSON.parse(JSON.stringify(graph));

  for (const [nodeIdStr, widgetPatches] of Object.entries(patch)) {
    const nodeId = Number(nodeIdStr);
    const node = cloned.nodes.find((n) => n.id === nodeId);
    if (!node) {
      console.warn(`[templateInjector] patch targets unknown node ${nodeId}`);
      continue;
    }
    if (!node.widgets_values) node.widgets_values = [];
    for (const [indexStr, value] of Object.entries(widgetPatches)) {
      node.widgets_values[Number(indexStr)] = value;
    }
  }

  return cloned;
}

/**
 * Convert a ComfyUI workflow graph to the API prompt format.
 *
 * The API prompt format is a flat map of `{ nodeId: { class_type, inputs } }`
 * where inputs are either literal values (widget values) or `[nodeId, slot]`
 * tuples (links from another node's output).
 *
 * Bypassed nodes (mode === 4) and purely visual nodes are excluded.
 *
 * For node types listed in WIDGET_DEFS, widget values are mapped to named
 * inputs in declaration order (Path A). For other node types, only inputs
 * that appear in the graph's `inputs` array are processed — this covers linked
 * inputs and connectable widgets, but misses pure widget values (Path B).
 * Path B is still correct for nodes whose pure widget inputs are either
 * optional or have server-side defaults that ComfyUI applies automatically.
 */
function graphToApiPrompt(graph: WorkflowGraph): ComfyWorkflow {
  // Build link lookup: linkId → [fromNodeId, fromSlot]
  const linkMap = new Map<number, [string, number]>();
  for (const [linkId, fromNodeId, fromSlot] of graph.links) {
    linkMap.set(linkId, [String(fromNodeId), fromSlot]);
  }

  const prompt: ComfyWorkflow = {};

  for (const node of graph.nodes) {
    if (node.mode === 4) continue; // bypassed
    if (SKIP_NODE_TYPES.has(node.type)) continue;

    const apiInputs: Record<string, unknown> = {};
    const widgetDefs = WIDGET_DEFS[node.type];

    if (widgetDefs) {
      // ── Path A: explicit widget definitions ─────────────────────────────
      // Step 1: Process pure nodal link inputs (no widget property on the slot).
      for (const input of node.inputs ?? []) {
        if (!input.widget && input.link != null) {
          const ref = linkMap.get(input.link);
          if (ref) apiInputs[input.name] = ref;
        }
      }

      // Step 2: Walk WIDGET_DEFS in order, consuming widgets_values slots.
      let widgetIndex = 0;
      for (const def of widgetDefs) {
        if (def.hidden) {
          // Consume the slot but don't emit to the API prompt.
          widgetIndex++;
          continue;
        }
        if (def.connectable) {
          // Widget that can accept a link — check whether it's currently linked.
          const linked = node.inputs?.find(
            (i) =>
              (i.name === def.name || i.widget?.name === def.name) &&
              i.link != null,
          );
          if (linked) {
            const ref = linkMap.get(linked.link!);
            if (ref) apiInputs[def.name] = ref;
          } else {
            apiInputs[def.name] = node.widgets_values?.[widgetIndex];
          }
          widgetIndex++;
        } else {
          // Pure widget — always reads from widgets_values.
          apiInputs[def.name] = node.widgets_values?.[widgetIndex];
          widgetIndex++;
        }
      }
    } else {
      // ── Path B: iterate inputs array only ────────────────────────────────
      // Handles linked inputs and connectable widgets that appear in the
      // graph's inputs array. Pure widget values that don't appear there are
      // NOT included — this is acceptable for nodes whose remaining widgets
      // are optional or have ComfyUI-side defaults.
      let widgetIndex = 0;

      for (const input of node.inputs ?? []) {
        if (input.link != null) {
          // Input is linked — use [fromNodeId, fromSlot] reference.
          const ref = linkMap.get(input.link);
          if (ref) apiInputs[input.name] = ref;
          // If it also has a widget, that slot in widgets_values is consumed.
          if (input.widget) {
            widgetIndex++;
          }
        } else if (input.widget) {
          // Connectable widget that is currently unlinked — use the widget value.
          apiInputs[input.name] = node.widgets_values?.[widgetIndex];
          widgetIndex++;
        }
        // Unlinked, non-widget inputs (disconnected optional ports) are skipped.
      }
    }

    prompt[String(node.id)] = {
      class_type: node.type,
      inputs: apiInputs,
    };
  }

  return prompt;
}

// ─── Public API ──────────────────────────────────────────────────────────────

export interface InjectedTemplate {
  /** API prompt format — pass as `prompt` in the ComfyUI /prompt body. */
  apiPrompt: ComfyWorkflow;
  /**
   * Full patched workflow graph — pass as
   * `extra_data.extra_pnginfo.workflow` so output PNGs embed the workflow.
   */
  workflow: WorkflowGraph;
}

/**
 * Patch a workflow template with user parameters and convert to API format.
 *
 * @param templateJson  Raw workflow graph JSON (import the .json file directly).
 * @param patch         Widget value overrides keyed by nodeId → widgetIndex.
 * @returns             `{ apiPrompt, workflow }` ready to submit.
 *
 * @example
 * ```ts
 * import animaTemplate from './templates/workflows/anima-guy90s-v10.json';
 * import { ANIMA_PATCH } from './templates/anima';
 *
 * const { apiPrompt, workflow } = injectTemplate(animaTemplate, ANIMA_PATCH({
 *   positivePrompt: 'masterpiece, best quality',
 *   seed: 42,
 * }));
 *
 * await comfyClient.submitPrompt({ prompt: apiPrompt, workflow });
 * ```
 */
export function injectTemplate(
  templateJson: WorkflowGraph,
  patch: WorkflowPatch,
): InjectedTemplate {
  const patched = applyPatch(templateJson, patch);
  const apiPrompt = graphToApiPrompt(patched);
  return { apiPrompt, workflow: patched };
}
