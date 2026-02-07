/**
 * Next.js API Route: POST /api/dataset/bulk-tag-operation
 * Perform bulk tag operations on a dataset
 *
 * Migrated from Python FastAPI: api/routes/dataset.py -> bulk_tag_operation
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

const ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'];

interface BulkTagOperationRequest {
  dataset_path: string;
  operation: 'add' | 'remove' | 'replace';
  tags: string[];
  replace_with?: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: BulkTagOperationRequest = await request.json();
    const { dataset_path, operation, tags, replace_with } = body;

    if (!dataset_path) {
      return NextResponse.json(
        { error: 'Missing dataset_path' },
        { status: 400 }
      );
    }

    if (!operation || !['add', 'remove', 'replace'].includes(operation)) {
      return NextResponse.json(
        { error: 'Invalid operation. Must be add, remove, or replace' },
        { status: 400 }
      );
    }

    if (!Array.isArray(tags) || tags.length === 0) {
      return NextResponse.json(
        { error: 'tags must be a non-empty array' },
        { status: 400 }
      );
    }

    // Validate dataset path exists
    try {
      const stats = await fs.stat(dataset_path);
      if (!stats.isDirectory()) {
        return NextResponse.json(
          { error: 'Path is not a directory' },
          { status: 400 }
        );
      }
    } catch {
      return NextResponse.json(
        { error: `Dataset not found: ${dataset_path}` },
        { status: 404 }
      );
    }

    // Get all image files
    const entries = await fs.readdir(dataset_path, { withFileTypes: true });
    const imageFiles = entries
      .filter(entry => {
        if (!entry.isFile()) return false;
        const ext = path.extname(entry.name).toLowerCase();
        return ALLOWED_IMAGE_EXTENSIONS.includes(ext);
      })
      .map(entry => path.join(dataset_path, entry.name));

    let filesModified = 0;
    const errors: string[] = [];

    // Convert tags to lowercase set for case-insensitive matching
    const tagsLower = new Set(tags.map(t => t.toLowerCase().trim()));

    for (const imagePath of imageFiles) {
      try {
        const captionPath = imagePath.replace(path.extname(imagePath), '.txt');

        // Read existing caption
        let existingTags: string[] = [];
        try {
          const captionText = await fs.readFile(captionPath, 'utf-8');
          existingTags = captionText
            .trim()
            .split(',')
            .map(tag => tag.trim())
            .filter(tag => tag.length > 0);
        } catch {
          // Caption file doesn't exist
        }

        let newTags: string[];

        switch (operation) {
          case 'add':
            // Add tags that don't already exist
            newTags = [...existingTags];
            for (const tag of tags) {
              if (!existingTags.some(t => t.toLowerCase() === tag.toLowerCase())) {
                newTags.push(tag);
              }
            }
            break;

          case 'remove':
            // Remove matching tags (case-insensitive)
            newTags = existingTags.filter(tag => !tagsLower.has(tag.toLowerCase()));
            break;

          case 'replace':
            // Replace matching tags with the replacement text
            newTags = existingTags.map(tag => {
              if (tagsLower.has(tag.toLowerCase())) {
                return replace_with || '';
              }
              return tag;
            }).filter(tag => tag.length > 0);
            break;

          default:
            newTags = existingTags;
        }

        // Write if changed
        if (JSON.stringify(newTags) !== JSON.stringify(existingTags)) {
          await fs.writeFile(captionPath, newTags.join(', '), 'utf-8');
          filesModified++;
        }
      } catch (error) {
        errors.push(
          `${path.basename(imagePath)}: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    }

    return NextResponse.json({
      success: true,
      message: `${operation} operation completed on ${filesModified} files`,
      operation,
      files_modified: filesModified,
      total_files: imageFiles.length,
      errors,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: errorMessage, detail: errorMessage },
      { status: 500 }
    );
  }
}
