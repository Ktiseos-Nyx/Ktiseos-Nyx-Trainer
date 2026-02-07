/**
 * Next.js API Route: GET /api/dataset/images-with-tags
 * List images in a dataset with their associated tag files
 *
 * Migrated from Python FastAPI: api/routes/dataset.py -> get_images_with_tags
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

const ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'];

interface ImageWithTags {
  image_path: string;
  image_name: string;
  tags: string[];
  has_tags: boolean;
  url?: string;
}

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

    // Validate dataset path exists
    try {
      const stats = await fs.stat(datasetPath);
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
    const entries = await fs.readdir(datasetPath, { withFileTypes: true });

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
      const imagePath = path.join(datasetPath, imageName);
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

    return NextResponse.json({
      images,
      total: images.length,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: errorMessage, detail: errorMessage },
      { status: 500 }
    );
  }
}
