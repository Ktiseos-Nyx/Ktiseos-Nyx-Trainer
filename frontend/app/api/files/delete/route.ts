/**
 * Next.js API Route: DELETE /api/files/delete
 * Delete a file or directory
 *
 * Migrated from Python FastAPI: api/routes/files.py -> delete_file
 */

import { NextRequest, NextResponse } from 'next/server';
import { fileService } from '@/lib/node-services/file-service';

export async function DELETE(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const path = searchParams.get('path');

    if (!path) {
      return NextResponse.json(
        { error: 'Missing path parameter' },
        { status: 400 }
      );
    }

    const result = await fileService.delete(path);

    if (!result.success) {
      return NextResponse.json(
        { error: result.error || 'Failed to delete' },
        { status: 500 }
      );
    }

    return NextResponse.json(result);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);

    let status = 500;
    if (errorMessage.includes('Access denied')) {
      status = 403;
    } else if (errorMessage.includes('not found')) {
      status = 404;
    }

    return NextResponse.json(
      { error: errorMessage, detail: errorMessage },
      { status }
    );
  }
}
