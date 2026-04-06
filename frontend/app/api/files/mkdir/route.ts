/**
 * Next.js API Route: POST /api/files/mkdir
 * Create a new directory
 *
 * Migrated from Python FastAPI: api/routes/files.py -> create_directory
 */

import { NextRequest, NextResponse } from 'next/server';
import { fileService } from '@/lib/node-services/file-service';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { path, name } = body;

    if (!path || !name) {
      return NextResponse.json(
        { error: 'Missing path or name parameter' },
        { status: 400 }
      );
    }

    const result = await fileService.createDirectory(path, name);

    if (!result.success) {
      const status = result.error?.includes('already exists') ? 400 : 500;

      return NextResponse.json(
        { error: result.error || 'Failed to create directory' },
        { status }
      );
    }

    return NextResponse.json(result);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);

    let status = 500;
    if (errorMessage.includes('Access denied')) {
      status = 403;
    }

    return NextResponse.json(
      { error: errorMessage, detail: errorMessage },
      { status }
    );
  }
}
