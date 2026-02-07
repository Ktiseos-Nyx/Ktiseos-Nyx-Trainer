/**
 * Next.js API Route: GET /api/utilities/datasets/browse
 * Browse available datasets in the datasets/ directory
 *
 * Migrated from Python FastAPI: api/routes/utilities.py -> browse_datasets
 */

import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

const ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'];

interface DatasetInfo {
  path: string;
  name: string;
  image_count: number;
  repeats: number;
  caption: string | null;
}

// Get datasets directory from environment or use default
function getDatasetsDir(): string {
  const candidates = [
    process.env.DATASETS_DIR,
    '/workspace/Ktiseos-Nyx-Trainer/datasets',
    path.join(process.cwd(), 'datasets'),
  ];

  for (const dir of candidates) {
    if (dir) return dir;
  }

  return path.join(process.cwd(), 'datasets');
}

/**
 * Extract Kohya parameters from folder name
 * Format: "N_caption" where N is repeat count
 */
function extractKohyaParams(folderName: string): { repeats: number; caption: string | null } {
  const match = folderName.match(/^(\d+)_(.+)$/);

  if (match) {
    return {
      repeats: parseInt(match[1], 10),
      caption: match[2].replace(/_/g, ' '),
    };
  }

  return {
    repeats: 1,
    caption: null,
  };
}

/**
 * Count images in a directory
 */
async function countImages(dirPath: string): Promise<number> {
  try {
    const entries = await fs.readdir(dirPath, { withFileTypes: true });
    return entries.filter(entry => {
      if (!entry.isFile()) return false;
      const ext = path.extname(entry.name).toLowerCase();
      return ALLOWED_IMAGE_EXTENSIONS.includes(ext);
    }).length;
  } catch {
    return 0;
  }
}

export async function GET() {
  try {
    const datasetsDir = getDatasetsDir();

    // Check if datasets directory exists
    try {
      await fs.stat(datasetsDir);
    } catch {
      // Create it if it doesn't exist
      await fs.mkdir(datasetsDir, { recursive: true });
      return NextResponse.json({ datasets: [] });
    }

    // Get all directories in datasets folder
    const entries = await fs.readdir(datasetsDir, { withFileTypes: true });
    const directories = entries.filter(entry => entry.isDirectory());

    if (directories.length === 0) {
      return NextResponse.json({ datasets: [] });
    }

    // Get info for each dataset
    const datasets: DatasetInfo[] = [];

    for (const dir of directories) {
      const datasetPath = path.join(datasetsDir, dir.name);
      const { repeats, caption } = extractKohyaParams(dir.name);
      const imageCount = await countImages(datasetPath);

      datasets.push({
        path: datasetPath,
        name: dir.name,
        image_count: imageCount,
        repeats,
        caption,
      });
    }

    // Sort by modification time (most recent first) - get stats for each
    const datasetsWithTime = await Promise.all(
      datasets.map(async (dataset) => {
        try {
          const stats = await fs.stat(dataset.path);
          return { ...dataset, mtime: stats.mtimeMs };
        } catch {
          return { ...dataset, mtime: 0 };
        }
      })
    );

    datasetsWithTime.sort((a, b) => b.mtime - a.mtime);

    // Remove mtime from response
    const result = datasetsWithTime.map(({ mtime, ...rest }) => rest);

    return NextResponse.json({ datasets: result });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: errorMessage, detail: errorMessage },
      { status: 500 }
    );
  }
}
