/**
 * GET /api/dataset/list
 * List all datasets with metadata.
 *
 * Previous bug: used path.join(projectRoot, 'dataset') (singular) which never
 * matched the actual on-disk 'datasets/' directory, causing the panel to always show empty.
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import { getDatasetsDir, IMAGE_EXTENSIONS } from '@/lib/node-services/datasets';

export const revalidate = 0;

const NO_STORE = { headers: { 'Cache-Control': 'no-store' } };

/**
 * List datasets found in the configured datasets directory and report per-dataset image and caption counts.
 */
export async function GET(_request: NextRequest) {
  try {
    const datasetRoot = getDatasetsDir();
    if (!datasetRoot) {
      return NextResponse.json({ datasets: [], total: 0 }, NO_STORE);
    }

    const entries = await fs.readdir(datasetRoot, { withFileTypes: true });
    const datasets: Array<{
      name: string;
      path: string;
      image_count: number;
      caption_count: number;
    }> = [];

    for (const entry of entries) {
      if (!entry.isDirectory()) continue;

      const datasetPath = path.join(datasetRoot, entry.name);
      const files = await fs.readdir(datasetPath);
      const lower = files.map(f => f.toLowerCase());

      datasets.push({
        name: entry.name,
        path: datasetPath,
        image_count: lower.filter(f => IMAGE_EXTENSIONS.some(ext => f.endsWith(ext))).length,
        caption_count: lower.filter(f => f.endsWith('.txt')).length,
      });
    }

    return NextResponse.json({ datasets, total: datasets.length }, NO_STORE);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500, ...NO_STORE }
    );
  }
}
