/**
 * Next.js API Route: GET /api/utilities/directories
 * Get project directory paths (relative to trainer root)
 *
 * Migrated from Python FastAPI: api/routes/utilities.py -> get_directories
 */

import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    output: 'output',
    datasets: 'datasets',
    pretrained_model: 'pretrained_model',
    vae: 'vae',
  });
}
