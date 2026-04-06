/**
 * Next.js API Route: GET /api/files/read
 * Read a text file
 *
 * Migrated from Python FastAPI: api/routes/files.py -> read_file
 */

import { NextRequest, NextResponse } from 'next/server';
import { fileService } from '@/lib/node-services/file-service';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const path = searchParams.get('path');

    if (!path) {
      return NextResponse.json(
        { error: 'Missing path parameter' },
        { status: 400 }
      );
    }

    const result = await fileService.readFile(path);
    return NextResponse.json(result);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);

    let status = 500;
    if (errorMessage.includes('Access denied')) {
      status = 403;
    } else if (errorMessage.includes('not found')) {
      status = 404;
    } else if (errorMessage.includes('Not a file')) {
      status = 400;
    }

    return NextResponse.json(
      { error: errorMessage, detail: errorMessage },
      { status }
    );
  }
}
