/**
 * Next.js API Route: GET /api/utilities/directories
 * Get project directory paths (absolute filesystem paths, matching
 * the Python FastAPI response format).
 *
 * Migrated from Python FastAPI: api/routes/utilities.py -> get_directories
 */

import path from 'path';
import { NextResponse } from 'next/server';
import { getComfyuiModelDirs } from '@/lib/node-services/comfyui-paths';

const PROJECT_ROOT = path.join(/*turbopackIgnore: true*/ process.cwd(), '..');

function abs(sub: string): string {
  return path.resolve(PROJECT_ROOT, sub);
}

export async function GET() {
  const dirs: Record<string, string> = {
    output: abs('output'),
    datasets: abs('datasets'),
    pretrained_model: abs('pretrained_model'),
    vae: abs('vae'),
  };

  const comfyui = getComfyuiModelDirs();
  for (const [key, value] of Object.entries(comfyui)) {
    dirs[key] = value;
  }

  return NextResponse.json(dirs);
}
