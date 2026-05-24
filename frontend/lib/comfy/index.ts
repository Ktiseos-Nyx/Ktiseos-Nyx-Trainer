/**
 * ComfyUI library — public API.
 *
 * Import from here in application code:
 *   import { comfyClient, useComfyConnection, buildTxt2ImgWorkflow } from '@/lib/comfy';
 */

export { comfyClient } from './client';
export type { ComfyWsHandle, ComfyWsOptions } from './client';

export { useComfyConnection } from './useComfyConnection';
export type { UseComfyConnectionReturn } from './useComfyConnection';

export { buildTxt2ImgWorkflow, buildImg2ImgWorkflow } from './workflows';
export type { Txt2ImgParams, Img2ImgParams, LoraEntry } from './workflows';

export type {
  ComfyWorkflow,
  ComfyNode,
  ComfyPromptRequest,
  ComfyPromptResponse,
  ComfyQueueStatus,
  ComfyHistory,
  ComfyHistoryEntry,
  ComfyOutputFile,
  ComfyOutputs,
  ComfySystemStats,
  ComfyObjectInfo,
  ComfyConnectionState,
  ComfyConnectionStatus,
  ComfyWsMessage,
} from './types';
