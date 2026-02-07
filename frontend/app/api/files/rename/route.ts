/**
 * Next.js API Route: POST /api/files/rename
 * Rename a file or directory
 *
 * Migrated from Python FastAPI: api/routes/files.py -> rename_file
 */

import { NextRequest, NextResponse } from 'next/server';
import { fileService } from '@/lib/node-services/file-service';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { old_path, new_name } = body;

    if (!old_path || !new_name) {
      return NextResponse.json(
        { error: 'Missing old_path or new_name parameter' },
        { status: 400 }
      );
    }

    const result = await fileService.rename(old_path, new_name);

    if (!result.success) {
      const status = result.error?.includes('not found') ? 404
                   : result.error?.includes('already exists') ? 400
                   : 500;

      return NextResponse.json(
        { error: result.error || 'Failed to rename' },
        { status }
      );
    }

    return NextResponse.json({
      success: true,
      old_path,
      new_path: result.path,
    });
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
