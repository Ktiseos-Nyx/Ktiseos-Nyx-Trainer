/**
 * Next.js API Route: POST /api/dataset/upload-batch
 * Upload multiple files to a dataset
 *
 * Migrated from Python FastAPI: api/routes/dataset.py -> upload_batch
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

// Get datasets directory from environment or use default
function getDatasetsDir(): string {
  // Check common paths
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

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const files = formData.getAll('files');
    const datasetName = formData.get('dataset_name');

    if (!datasetName || typeof datasetName !== 'string') {
      return NextResponse.json(
        { error: 'Missing dataset_name' },
        { status: 400 }
      );
    }

    if (!files || files.length === 0) {
      return NextResponse.json(
        { error: 'No files provided' },
        { status: 400 }
      );
    }

    // Sanitize dataset name
    const safeName = datasetName.replace(/[^a-zA-Z0-9_-]/g, '_');
    const datasetPath = path.join(getDatasetsDir(), safeName);

    // Create dataset directory if it doesn't exist
    await fs.mkdir(datasetPath, { recursive: true });

    const uploaded: string[] = [];
    const errors: string[] = [];

    for (const file of files) {
      if (!(file instanceof File)) {
        errors.push('Invalid file in upload');
        continue;
      }

      try {
        // Sanitize filename
        const safeFilename = file.name.replace(/[^a-zA-Z0-9._-]/g, '_');
        const filePath = path.join(datasetPath, safeFilename);

        // Read file content as ArrayBuffer and convert to Buffer
        const arrayBuffer = await file.arrayBuffer();
        const buffer = Buffer.from(arrayBuffer);

        // Write file
        await fs.writeFile(filePath, buffer);
        uploaded.push(safeFilename);
      } catch (error) {
        errors.push(
          `${file.name}: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    }

    return NextResponse.json({
      success: true,
      message: `Uploaded ${uploaded.length} files to ${safeName}`,
      dataset_name: safeName,
      dataset_path: datasetPath,
      files_uploaded: uploaded.length,
      uploaded_files: uploaded,
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

// Increase body size limit for file uploads
export const config = {
  api: {
    bodyParser: false,
  },
};
