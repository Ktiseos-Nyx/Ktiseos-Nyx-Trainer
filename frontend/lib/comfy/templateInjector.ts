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

/**
 * Per-node-type count of hidden "control" widget values that precede the
 * listed input names in `widgets_values` but are NOT listed in `inputs`.
 *
 * ComfyUI embeds UI-only controls (e.g. `control_after_generate` after `seed`
 * in KSampler) inside `widgets_values` without listing them in the node's
 * `inputs` array. We must skip them to correctly index subsequent widget values.
 *
 * Format: input name → number of hidden widgets that appear AFTER this input
 * in `widgets_values` before the next named input begins.
 */
const HIDDEN_WIDGETS_AFTER: Record<string, Record<string, number>> = {
  KSampler: { seed: 1 },          // control_after_generate
  KSamplerAdvanced: { noise_seed: 1 },
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
    let widgetIndex = 0;
    const hiddenAfter = HIDDEN_WIDGETS_AFTER[node.type] ?? {};

    for (const input of node.inputs ?? []) {
      if (input.link != null) {
        // Input is linked — use [fromNodeId, fromSlot] reference
        const ref = linkMap.get(input.link);
        if (ref) apiInputs[input.name] = ref;
        // If it also has a widget, that slot in widgets_values is still consumed
        if (input.widget) {
          widgetIndex++;
          widgetIndex += hiddenAfter[input.name] ?? 0;
        }
      } else if (input.widget) {
        // Input is a plain widget — use the stored value
        apiInputs[input.name] = node.widgets_values?.[widgetIndex];
        widgetIndex++;
        widgetIndex += hiddenAfter[input.name] ?? 0;
      }
      // Unlinked, non-widget inputs (disconnected optional ports) are skipped
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
