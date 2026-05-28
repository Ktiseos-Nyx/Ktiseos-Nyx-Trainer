/**
 * ComfyUI workflow templates — public re-export.
 *
 * Usage:
 *   import { buildAnimaPatch, buildSdxlKnxPatch } from '@/lib/comfy/templates';
 *   import { injectTemplate } from '@/lib/comfy/templateInjector';
 *   import animaWorkflow from '@/lib/comfy/templates/workflows/anima-guy90s-v10.json';
 *   import sdxlWorkflow  from '@/lib/comfy/templates/workflows/sdxl-knx-v1.json';
 */

export { buildAnimaPatch } from './anima';
export type { AnimaTemplateParams } from './anima';

export { buildSdxlKnxPatch } from './sdxl-knx';
export type { SdxlKnxTemplateParams } from './sdxl-knx';
