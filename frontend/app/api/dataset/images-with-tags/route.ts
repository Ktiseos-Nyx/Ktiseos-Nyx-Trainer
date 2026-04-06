/**
 * Next.js API Route: GET /api/dataset/images-with-tags
 * List images in a dataset with their associated tag files
 *
 * Migrated from Python FastAPI: api/routes/dataset.py -> get_images_with_tags
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import { validateDatasetPath, DATASETS_DIR } from '@/lib/node-services/path-validation';
import { IMAGE_EXTENSIONS } from '@/lib/node-services/datasets';

// Opt out of Next.js static caching so the tag editor reflects saves immediately.
export const revalidate = 0;

interface ImageWithTags {
  image_path: string;
  image_name: string;
  tags: string[];
  has_tags: boolean;
}

const NO_STORE = { headers: { 'Cache-Control': 'no-store' } };

function jsonNoStore(body: Record<string, unknown>, status = 200) {
  return NextResponse.json(body, { status, ...NO_STORE });
}

/**
 * List images in a dataset directory and include parsed tags from accompanying `.txt` files.
 *
 * Expects a `dataset_path` query parameter. For each image file in the resolved dataset directory,
 * attempts to read a sibling `.txt` caption file, splits its contents by commas, trims tokens,
 * and sets `has_tags` to `true` when any non-empty tags are found.
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const datasetPath = searchParams.get('dataset_path');

    if (!datasetPath) {
      return jsonNoStore({ error: 'Missing dataset_path parameter' }, 400);
    }

    const trimmedPath = datasetPath.trim();
    if (!trimmedPath || /^\.+$/.test(trimmedPath)) {
      return jsonNoStore({ error: 'Invalid dataset_path parameter' }, 400);
    }

    // If just a bare name (no path separators), resolve under datasets dir
    // Try datasets/ (plural) first, then dataset/ (singular)
    let fullPath: string;
    if (trimmedPath.includes('/') || trimmedPath.includes('\\')) {
      fullPath = trimmedPath;
    } else {
      const pluralPath = path.join(DATASETS_DIR, trimmedPath);
      const singularPath = path.join(path.resolve(DATASETS_DIR, '..', 'dataset'), trimmedPath);
      try {
        await fs.stat(pluralPath);
        fullPath = pluralPath;
      } catch {
        try {
          await fs.stat(singularPath);
          fullPath = singularPath;
        } catch {
          // Fall through — validateDatasetPath will catch it
          fullPath = pluralPath;
        }
      }
    }

    // Security: confine to datasets directory
    let resolvedPath: string;
    try {
      resolvedPath = validateDatasetPath(fullPath);
    } catch {
      return jsonNoStore({ error: 'Access denied: path outside allowed directories' }, 403);
    }

    // Validate dataset path exists
    try {
      const stats = await fs.stat(resolvedPath);
      if (!stats.isDirectory()) {
        return jsonNoStore({ error: 'Path is not a directory' }, 400);
      }
    } catch {
      return jsonNoStore({ error: `Dataset not found: ${datasetPath}` }, 404);
    }

    // Get all files in directory
    const entries = await fs.readdir(resolvedPath, { withFileTypes: true });

    // Filter to image files
    const imageFiles = entries
      .filter(entry => {
        if (!entry.isFile()) return false;
        const ext = path.extname(entry.name).toLowerCase();
        return IMAGE_EXTENSIONS.includes(ext);
      })
      .map(entry => entry.name);

    // Build list of images with their tags
    const images: ImageWithTags[] = [];

    for (const imageName of imageFiles) {
      const imagePath = path.join(resolvedPath, imageName);
      const captionPath = imagePath.replace(path.extname(imagePath), '.txt');

      let tags: string[] = [];
      let hasTags = false;

      try {
        const captionText = await fs.readFile(captionPath, 'utf-8');
        tags = captionText
          .trim()
          .split(',')
          .map(tag => tag.trim())
          .filter(tag => tag.length > 0);
        hasTags = tags.length > 0;
      } catch {
        // Caption file doesn't exist
      }

      images.push({
        image_path: imagePath,
        image_name: imageName,
        tags,
        has_tags: hasTags,
      });
    }

    // Sort by name
    images.sort((a, b) => a.image_name.localeCompare(b.image_name));

    return jsonNoStore({ images, total: images.length });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return jsonNoStore({ error: errorMessage, detail: errorMessage }, 500);
  }
}
