/**
 * ComfyUI Model Directory Resolution
 *
 * Mirrors the Python backend's `_comfyui_model_dirs()` and
 * `get_comfyui_models_path()` from api/routes/utilities.py and settings.py.
 *
 * Used by the Next.js directories route and anywhere ComfyUI model
 * subdirectories need to be listed.
 */

import fs from 'fs';
import path from 'path';

const PROJECT_ROOT = path.join(/*turbopackIgnore: true*/ process.cwd(), '..');

/**
 * Resolve ComfyUI's models directory using the same priority order as
 * the Python backend:
 *   1. {project_root}/ComfyUI/models (bundled — authoritative if it exists)
 *   2. COMFYUI_MODELS_PATH env var
 *   3. comfyui_models_path from user_settings.json
 *   4. {project_root}/ComfyUI/models (fallback string)
 */
export function getComfyuiModelsPath(): string | null {
  const anchored = path.resolve(PROJECT_ROOT, 'ComfyUI', 'models');

  if (fs.existsSync(anchored) && fs.statSync(anchored).isDirectory()) {
    return anchored;
  }

  const envPath = process.env.COMFYUI_MODELS_PATH;
  if (envPath && fs.existsSync(envPath) && fs.statSync(envPath).isDirectory()) {
    return envPath;
  }

  try {
    const settingsDir = path.join(PROJECT_ROOT, 'user_config');
    const settingsFile = path.join(settingsDir, 'user_settings.json');
    if (fs.existsSync(settingsFile)) {
      const settings = JSON.parse(fs.readFileSync(settingsFile, 'utf-8'));
      const settingsPath = settings.comfyui_models_path;
      if (
        settingsPath &&
        fs.existsSync(settingsPath) &&
        fs.statSync(settingsPath).isDirectory()
      ) {
        return settingsPath;
      }
    }
  } catch {
    // Settings file missing or unreadable — use fallback
  }

  return anchored;
}

/**
 * Resolve ComfyUI's standard model subdirectories.
 *
 * Scans the base-model folders (checkpoints/, diffusion_models/, unet/)
 * plus loras/ under the ComfyUI models root. Returns only the directories
 * that actually exist on disk.
 *
 * If none of the standard subfolders exist, the root itself is returned
 * as a fallback so users on custom setups still see files.
 */
export function getComfyuiModelDirs(): Record<
  'comfyui_loras' | 'comfyui_checkpoints' | 'comfyui_diffusion_models' | 'comfyui_unet',
  string
> {
  const result: Record<
    'comfyui_loras' | 'comfyui_checkpoints' | 'comfyui_diffusion_models' | 'comfyui_unet',
    string
  > = {} as Record<string, string>;

  const base = getComfyuiModelsPath();
  if (!base) return result;

  let found = false;
  const subs: Array<[keyof typeof result, string]> = [
    ['comfyui_loras', 'loras'],
    ['comfyui_checkpoints', 'checkpoints'],
    ['comfyui_diffusion_models', 'diffusion_models'],
    ['comfyui_unet', 'unet'],
  ];

  for (const [key, sub] of subs) {
    const candidate = path.join(base, sub);
    if (fs.existsSync(candidate) && fs.statSync(candidate).isDirectory()) {
      result[key] = candidate;
      found = true;
    }
  }

  if (!found && fs.existsSync(base) && fs.statSync(base).isDirectory()) {
    result['comfyui_checkpoints'] = base;
    result['comfyui_loras'] = base;
  }

  return result;
}
