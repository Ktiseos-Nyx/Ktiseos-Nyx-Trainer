/**
 * Next.js API Route: GET /api/models/list
 * List downloaded models and VAEs
 *
 * Migrated from Python FastAPI: api/routes/models.py -> list_models_and_vaes
 */

import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import { settingsService } from '@/lib/node-services/settings-service';

interface ModelFile {
  name: string;
  path: string;
  size_bytes: number;
  size_mb: number;
  modified: number;
}

// Get model directories from environment or use defaults
function getModelDirs(): { modelDir: string; vaeDir: string; loraDir: string } {
  const baseDir = process.env.WORKSPACE_DIR || process.cwd();

  return {
    modelDir: process.env.PRETRAINED_MODEL_DIR || path.join(baseDir, 'pretrained_model'),
    vaeDir: process.env.VAE_DIR || path.join(baseDir, 'vae'),
    loraDir: process.env.OUTPUT_DIR || path.join(baseDir, 'output'),
  };
}

// Common model subdirectories on cloud GPU providers (RunPod, VastAI, etc.)
const CLOUD_MODEL_PATHS = [
  '/workspace/models',
  '/workspace/models/Stable-diffusion',
  '/workspace/models/checkpoints',
  '/workspace/models/unet',
  '/workspace/ComfyUI/models/checkpoints',
];

const CLOUD_VAE_PATHS = [
  '/workspace/models/vae',
  '/workspace/models/VAE',
  '/workspace/ComfyUI/models/vae',
];

const CLOUD_TEXT_ENCODER_PATHS = [
  '/workspace/models/clip',
  '/workspace/models/CLIP',
  '/workspace/models/t5',
  '/workspace/models/text_encoders',
  '/workspace/models/text_encoder',
  '/workspace/ComfyUI/models/clip',
  '/workspace/ComfyUI/models/text_encoders',
];

const MODEL_EXTENSIONS = ['.safetensors', '.ckpt', '.pt', '.pth', '.bin'];

/**
 * List files in a directory with size information
 */
async function listModelFiles(dirPath: string): Promise<ModelFile[]> {
  const files: ModelFile[] = [];

  try {
    const stats = await fs.stat(dirPath);
    if (!stats.isDirectory()) return files;
  } catch {
    // Directory doesn't exist
    return files;
  }

  const entries = await fs.readdir(dirPath, { withFileTypes: true });

  for (const entry of entries) {
    if (!entry.isFile()) continue;

    const ext = path.extname(entry.name).toLowerCase();
    if (!MODEL_EXTENSIONS.includes(ext)) continue;

    const filePath = path.join(dirPath, entry.name);

    try {
      const stats = await fs.stat(filePath);
      files.push({
        name: entry.name,
        path: filePath,
        size_bytes: stats.size,
        size_mb: Math.round((stats.size / (1024 * 1024)) * 100) / 100,
        modified: stats.mtimeMs,
      });
    } catch {
      // Skip files we can't stat
    }
  }

  // Sort by name
  files.sort((a, b) => a.name.localeCompare(b.name));

  return files;
}

/**
 * Canonicalize a path for deduplication — resolves symlinks via realpath,
 * falling back to path.resolve if the file isn't accessible.
 */
async function canonicalizePath(filePath: string): Promise<string> {
  try {
    return await fs.realpath(filePath);
  } catch {
    return path.resolve(filePath);
  }
}

/**
 * Merge model file arrays, deduplicating by canonical (realpath) path so
 * symlinked directories pointing to the same physical file don't produce
 * duplicate entries.
 */
async function mergeModelFiles(primary: ModelFile[], ...extras: ModelFile[][]): Promise<ModelFile[]> {
  const seen = new Set<string>(
    await Promise.all(primary.map((f) => canonicalizePath(f.path)))
  );
  const merged = [...primary];
  for (const extra of extras) {
    for (const file of extra) {
      const canonical = await canonicalizePath(file.path);
      if (!seen.has(canonical)) {
        seen.add(canonical);
        merged.push(file);
      }
    }
  }
  merged.sort((a, b) => a.name.localeCompare(b.name));
  return merged;
}

export async function GET() {
  try {
    const { modelDir, vaeDir, loraDir } = getModelDirs();
    const { extra_model_dirs, extra_vae_dirs } = await settingsService.getExtraModelDirs();

    const allModelDirs = [modelDir, ...CLOUD_MODEL_PATHS, ...extra_model_dirs];
    const allVaeDirs = [vaeDir, ...CLOUD_VAE_PATHS, ...extra_vae_dirs];
    // Text encoders: scan dedicated dirs + the main model dir (some setups mix them)
    const allTextEncoderDirs = [...CLOUD_TEXT_ENCODER_PATHS, modelDir];

    const allResults = await Promise.all([
      ...allModelDirs.map((d) => listModelFiles(d)),
      ...allVaeDirs.map((d) => listModelFiles(d)),
      ...allTextEncoderDirs.map((d) => listModelFiles(d)),
      listModelFiles(loraDir),
    ]);

    const modelResults = allResults.slice(0, allModelDirs.length);
    const vaeResults = allResults.slice(allModelDirs.length, allModelDirs.length + allVaeDirs.length);
    const textEncoderResults = allResults.slice(
      allModelDirs.length + allVaeDirs.length,
      allModelDirs.length + allVaeDirs.length + allTextEncoderDirs.length
    );
    const loras = allResults[allResults.length - 1];

    const [mergedModels, mergedVaes, mergedTextEncoders] = await Promise.all([
      mergeModelFiles(modelResults[0], ...modelResults.slice(1)),
      mergeModelFiles(vaeResults[0], ...vaeResults.slice(1)),
      mergeModelFiles(textEncoderResults[0], ...textEncoderResults.slice(1)),
    ]);

    return NextResponse.json({
      success: true,
      models: mergedModels,
      vaes: mergedVaes,
      text_encoders: mergedTextEncoders,
      loras,
      model_dir: modelDir,
      vae_dir: vaeDir,
      lora_dir: loraDir,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: errorMessage, detail: errorMessage },
      { status: 500 }
    );
  }
}
