/**
 * Next.js API Route: GET /api/dataset/images-with-tags
 * List images in a dataset with their associated tag files
 *
 * Migrated from Python FastAPI: api/routes/dataset.py -> get_images_with_tags
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import { validateDatasetPath } from '@/lib/node-services/path-validation';

const ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'];

// Opt out of Next.js static caching so the tag editor reflects saves immediately.
export const revalidate = 0;

interface ImageWithTags {
  image_path: string;
  image_name: string;
  tags: string[];
  has_tags: boolean;
  url?: string;
}

/**
 * List images in a dataset directory and include parsed tags from accompanying `.txt` files.
 *
 * Expects a `dataset_path` query parameter. For each image file in the resolved dataset directory, attempts to read a sibling `.txt` caption file, splits its contents by commas, trims tokens, and sets `has_tags` to `true` when any non-empty tags are found.
 *
 * @param request - Incoming request containing the `dataset_path` query parameter
 * @returns An object with:
 *  - `images`: an array of ImageWithTags objects (`image_path`, `image_name`, `tags`, `has_tags`, optional `url`) and
 *  - `total`: the number of images found
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const datasetPath = searchParams.get('dataset_path');

    if (!datasetPath) {
      return NextResponse.json(
        { error: 'Missing dataset_path parameter' },
        { status: 400 }
      );
    }

    // Security: confine to datasets directory
    let resolvedPath: string;
    try {
      resolvedPath = validateDatasetPath(datasetPath);
    } catch {
      return NextResponse.json(
        { error: 'Access denied: path outside allowed directories' },
        { status: 403 }
      );
    }

    // Validate dataset path exists
    try {
      const stats = await fs.stat(resolvedPath);
      if (!stats.isDirectory()) {
        return NextResponse.json(
          { error: 'Path is not a directory' },
          { status: 400 }
        );
      }
    } catch {
      return NextResponse.json(
        { error: `Dataset not found: ${datasetPath}` },
        { status: 404 }
      );
    }

    // Get all files in directory
    const entries = await fs.readdir(resolvedPath, { withFileTypes: true });

    // Filter to image files
    const imageFiles = entries
      .filter(entry => {
        if (!entry.isFile()) return false;
        const ext = path.extname(entry.name).toLowerCase();
        return ALLOWED_IMAGE_EXTENSIONS.includes(ext);
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

    return NextResponse.json(
      { images, total: images.length },
      { headers: { 'Cache-Control': 'no-store' } }
    );
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: errorMessage, detail: errorMessage },
      { status: 500 }
    );
  }
}
