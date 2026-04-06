/**
 * Next.js API Route: POST /api/files/write
 * Write a text file
 *
 * Migrated from Python FastAPI: api/routes/files.py -> write_file
 */

import { NextRequest, NextResponse } from 'next/server';
import { fileService } from '@/lib/node-services/file-service';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { path, content } = body;

    if (!path) {
      return NextResponse.json(
        { error: 'Missing path parameter' },
        { status: 400 }
      );
    }

    if (content === undefined) {
      return NextResponse.json(
        { error: 'Missing content parameter' },
        { status: 400 }
      );
    }

    const result = await fileService.writeFile(path, content);

    if (!result.success) {
      return NextResponse.json(
        { error: result.error || 'Failed to write file' },
        { status: 500 }
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
