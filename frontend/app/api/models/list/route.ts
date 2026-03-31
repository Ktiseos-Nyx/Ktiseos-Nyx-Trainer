/**
 * Next.js API Route: GET /api/models/list
 * List downloaded models and VAEs
 *
 * Migrated from Python FastAPI: api/routes/models.py -> list_models_and_vaes
 */

import { NextResponse } from 'next/server';
import fsSync from 'fs';              // 👈 Sync fs for existsSync
import fs from 'fs/promises';         // 👈 Async fs for stat/readdir (renamed from default)
import path from 'path';
import { settingsService } from '@/lib/node-services/settings-service';

interface ModelFile {
  name: string;
  path: string;
  size_bytes: number;
  size_mb: number;
  modified: number;
}

/**
 * Smart project root detection
 * Works on Vast, RunPod, Local, Docker without breaking any environment
 */
function getProjectRoot(): string {
  // 1. Env var override (cloud deployments) — validate before trusting
  for (const envVar of ['WORKSPACE_DIR', 'PROJECT_ROOT']) {
    const envVal = process.env[envVar];
    if (!envVal) continue;
    try {
      if (fsSync.existsSync(path.join(envVal, 'pretrained_model'))) {
        return envVal;
      }
    } catch {}
    // Env var set but marker not found — fall through to auto-detection
  }

  // 2. Smart detection: prefer parent if it has the pretrained_model dir
  //    (Next.js cwd is frontend/, but models live in the parent)
  const cwd = process.cwd();
  const parent = path.resolve(cwd, '..');

  try {
    if (fsSync.existsSync(path.join(parent, 'pretrained_model'))) {
      return parent;
    }
  } catch {}

  // Check if cwd itself has pretrained_model (running from project root)
  try {
    if (fsSync.existsSync(path.join(cwd, 'pretrained_model'))) {
      return cwd;
    }
  } catch {}

  // 3. Fallback: if parent has package.json, it's likely the monorepo root
  try {
    if (fsSync.existsSync(path.join(parent, 'package.json'))) {
      return parent;
    }
  } catch {}

  return cwd;
}

// Get model directories from environment or use defaults
function getModelDirs(projectRoot: string): { modelDir: string; vaeDir: string; loraDir: string } {
  return {
    modelDir: process.env.PRETRAINED_MODEL_DIR || path.join(projectRoot, 'pretrained_model'),
    vaeDir: process.env.VAE_DIR || path.join(projectRoot, 'vae'),
    loraDir: process.env.OUTPUT_DIR || path.join(projectRoot, 'output'),
  };
}

/**
 * Common model subdirectories RELATIVE to project root
 * These are scanned in addition to the main pretrained_model/ directory
 * Works on Vast, RunPod, Local, Docker without hardcoding absolute paths
 */
// Extra model dirs to scan (relative to project root).
// NOTE: pretrained_model/ and vae/ are already scanned as modelDir/vaeDir
// via getModelDirs() — don't duplicate them here.
const RELATIVE_MODEL_PATHS = [
  'models/Stable-diffusion',    // Common ComfyUI/SD-webui layout
  'models/checkpoints',         // Alternative ComfyUI layout
  'models/unet',                // Some Flux setups
];

const RELATIVE_VAE_PATHS = [
  'models/vae',                 // ComfyUI layout
  'models/VAE',                 // Case variant
];

const RELATIVE_TEXT_ENCODER_PATHS = [
  'text_encoders',              // Default
  'models/clip',                // ComfyUI
  'models/CLIP',                // Case variant
  'models/t5',                  // T5-specific
];

const MODEL_EXTENSIONS = ['.safetensors', '.ckpt', '.pt', '.pth', '.bin'];

/**
 * List files in a directory with size information
 */
async function listModelFiles(dirPath: string): Promise<ModelFile[]> {
  const files: ModelFile[] = [];

  try {
    const stats = await fs.stat(dirPath); // 👈 Uses async 'fs' (fs/promises)
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
    return await fs.realpath(filePath); // 👈 Uses async 'fs'
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

/**
 * Helper: Resolve relative paths to absolute, filtering out non-existent dirs
 */
function resolveRelativePaths(relativePaths: string[], baseDir: string): string[] {
  return relativePaths
    .map(p => path.join(baseDir, p))
    .filter(p => {
      try {
        return fsSync.existsSync(p); // 👈 Uses sync fsSync
      } catch {
        return false;
      }
    });
}

export async function GET() {
  try {
    const projectRoot = getProjectRoot();
    const { modelDir, vaeDir, loraDir } = getModelDirs(projectRoot);
    const { extra_model_dirs, extra_vae_dirs } = await settingsService.getExtraModelDirs();

    // Resolve relative paths to absolute, filter out non-existent dirs
    const allModelDirs = [
      modelDir,
      ...resolveRelativePaths(RELATIVE_MODEL_PATHS, projectRoot),
      ...extra_model_dirs,
    ];
    
    const allVaeDirs = [
      vaeDir,
      ...resolveRelativePaths(RELATIVE_VAE_PATHS, projectRoot),
      ...extra_vae_dirs,
    ];
    
    const allTextEncoderDirs = [
      ...resolveRelativePaths(RELATIVE_TEXT_ENCODER_PATHS, projectRoot),
      modelDir, // Also scan main model dir for mixed setups
    ];

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
