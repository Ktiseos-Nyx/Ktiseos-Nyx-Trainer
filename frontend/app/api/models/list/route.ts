/**
 * Next.js API Route: GET /api/models/list
 * List downloaded models and VAEs
 *
 * Migrated from Python FastAPI: api/routes/models.py -> list_models_and_vaes
 */

import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

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

export async function GET() {
  try {
    const { modelDir, vaeDir, loraDir } = getModelDirs();

    const [models, vaes, loras] = await Promise.all([
      listModelFiles(modelDir),
      listModelFiles(vaeDir),
      listModelFiles(loraDir),
    ]);

    return NextResponse.json({
      success: true,
      models,
      vaes,
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
