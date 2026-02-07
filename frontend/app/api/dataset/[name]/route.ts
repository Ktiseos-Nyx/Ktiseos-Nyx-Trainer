/**
 * Next.js API Route: /api/dataset/[name]
 * Dataset operations by name
 *
 * DELETE - Delete a dataset
 * GET - Get dataset info
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

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

const ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'];

interface RouteParams {
  params: Promise<{ name: string }>;
}

/**
 * GET /api/dataset/[name]
 * Get dataset information
 */
export async function GET(request: NextRequest, { params }: RouteParams) {
  try {
    const { name } = await params;
    const datasetPath = path.join(getDatasetsDir(), name);

    // Check if dataset exists
    try {
      const stats = await fs.stat(datasetPath);
      if (!stats.isDirectory()) {
        return NextResponse.json(
          { error: 'Not a directory' },
          { status: 400 }
        );
      }
    } catch {
      return NextResponse.json(
        { error: `Dataset not found: ${name}` },
        { status: 404 }
      );
    }

    // Count images and captions
    const entries = await fs.readdir(datasetPath, { withFileTypes: true });
    let imageCount = 0;
    let captionCount = 0;
    let totalSize = 0;

    for (const entry of entries) {
      if (!entry.isFile()) continue;

      const ext = path.extname(entry.name).toLowerCase();
      const filePath = path.join(datasetPath, entry.name);

      try {
        const stats = await fs.stat(filePath);
        totalSize += stats.size;
      } catch {
        // Skip if can't stat
      }

      if (ALLOWED_IMAGE_EXTENSIONS.includes(ext)) {
        imageCount++;
      } else if (ext === '.txt') {
        captionCount++;
      }
    }

    return NextResponse.json({
      name,
      path: datasetPath,
      image_count: imageCount,
      caption_count: captionCount,
      total_size: totalSize,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: errorMessage, detail: errorMessage },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/dataset/[name]
 * Delete a dataset and all its contents
 */
export async function DELETE(request: NextRequest, { params }: RouteParams) {
  try {
    const { name } = await params;
    const datasetPath = path.join(getDatasetsDir(), name);

    // Check if dataset exists
    try {
      const stats = await fs.stat(datasetPath);
      if (!stats.isDirectory()) {
        return NextResponse.json(
          { error: 'Not a directory' },
          { status: 400 }
        );
      }
    } catch {
      return NextResponse.json(
        { error: `Dataset not found: ${name}` },
        { status: 404 }
      );
    }

    // Delete directory and all contents
    await fs.rm(datasetPath, { recursive: true, force: true });

    return NextResponse.json({
      success: true,
      message: `Dataset ${name} deleted successfully`,
      deleted_path: datasetPath,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: errorMessage, detail: errorMessage },
      { status: 500 }
    );
  }
}
