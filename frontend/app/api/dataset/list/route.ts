/**
 * GET /api/dataset/list
 * List all datasets with metadata
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

export async function GET(request: NextRequest) {
  try {
    const projectRoot = path.resolve(process.cwd(), '..');
    const datasetRoot = path.join(projectRoot, 'dataset');

    try {
      await fs.access(datasetRoot);
    } catch {
      return NextResponse.json({ datasets: [], total: 0 });
    }

    const entries = await fs.readdir(datasetRoot, { withFileTypes: true });
    const datasets = [];

    for (const entry of entries) {
      if (!entry.isDirectory()) continue;

      const datasetPath = path.join(datasetRoot, entry.name);
      
      // Count images
      const imageExtensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp'];
      const files = await fs.readdir(datasetPath);
      const images = files.filter(f => 
        imageExtensions.some(ext => f.toLowerCase().endsWith(ext))
      );
      
      // Count captions
      const captions = files.filter(f => f.endsWith('.txt'));

      datasets.push({
        name: entry.name,
        path: datasetPath,
        image_count: images.length,
        caption_count: captions.length,
      });
    }

    return NextResponse.json({
      datasets,
      total: datasets.length,
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
