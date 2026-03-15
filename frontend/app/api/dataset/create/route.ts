/**
 * POST /api/dataset/create
 * Create a new dataset directory
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import { getOrCreateDatasetsDir } from '@/lib/node-services/datasets';
import { validateDatasetPath } from '@/lib/node-services/path-validation';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { name } = body;

    if (!name) {
      return NextResponse.json(
        { error: 'Missing required field: name' },
        { status: 400 }
      );
    }

    // Sanitize dataset name
    const safeName = name.replace(/[^a-z0-9_-]/gi, '_');

    const datasetPath = path.join(getOrCreateDatasetsDir(), safeName);

    // Security: confine to datasets directory
    try {
      validateDatasetPath(datasetPath);
    } catch {
      return NextResponse.json(
        { error: 'Access denied: path outside allowed directories' },
        { status: 403 }
      );
    }

    // Check if already exists
    try {
      await fs.access(datasetPath);
      return NextResponse.json(
        { error: `Dataset already exists: ${safeName}` },
        { status: 400 }
      );
    } catch {
      // Doesn't exist, good to create
    }

    await fs.mkdir(datasetPath, { recursive: true });

    return NextResponse.json({
      success: true,
      name: safeName,
      path: datasetPath,
      message: `Dataset created: ${safeName}`,
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
