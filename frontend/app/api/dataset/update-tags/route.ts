/**
 * Next.js API Route: POST /api/dataset/update-tags
 * Update tags for a single image
 *
 * Migrated from Python FastAPI: api/routes/dataset.py -> update_tags
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

interface UpdateTagsRequest {
  image_path: string;
  tags: string[];
}

export async function POST(request: NextRequest) {
  try {
    const body: UpdateTagsRequest = await request.json();
    const { image_path, tags } = body;

    if (!image_path) {
      return NextResponse.json(
        { error: 'Missing image_path' },
        { status: 400 }
      );
    }

    if (!Array.isArray(tags)) {
      return NextResponse.json(
        { error: 'tags must be an array' },
        { status: 400 }
      );
    }

    // Verify image exists
    try {
      await fs.stat(image_path);
    } catch {
      return NextResponse.json(
        { error: `Image not found: ${image_path}` },
        { status: 404 }
      );
    }

    // Get caption path
    const captionPath = image_path.replace(path.extname(image_path), '.txt');

    // Join tags with comma separator
    const captionText = tags.join(', ');

    // Write caption file
    await fs.writeFile(captionPath, captionText, 'utf-8');

    return NextResponse.json({
      success: true,
      message: 'Tags updated successfully',
      image_path,
      caption_path: captionPath,
      tags_count: tags.length,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: errorMessage, detail: errorMessage },
      { status: 500 }
    );
  }
}
