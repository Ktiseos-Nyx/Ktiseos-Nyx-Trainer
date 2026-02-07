/**
 * Next.js API Route: GET /api/files/list
 * Lists directory contents
 *
 * Migrated from Python FastAPI: api/routes/files.py -> list_directory
 */

import { NextRequest, NextResponse } from 'next/server';
import { fileService } from '@/lib/node-services/file-service';

export async function GET(request: NextRequest) {
  try {
    // Get path from query parameters
    const searchParams = request.nextUrl.searchParams;
    const path = searchParams.get('path') || undefined;

    // Call file service
    const result = await fileService.listDirectory(path);

    return NextResponse.json(result);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);

    // Determine appropriate status code
    let status = 500;
    if (errorMessage.includes('Access denied')) {
      status = 403;
    } else if (errorMessage.includes('not found')) {
      status = 404;
    } else if (errorMessage.includes('not a directory')) {
      status = 400;
    }

    return NextResponse.json(
      {
        error: errorMessage,
        detail: errorMessage,
      },
      { status }
    );
  }
}
