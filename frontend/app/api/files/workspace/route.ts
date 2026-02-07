/**
 * Next.js API Route: GET /api/files/workspace
 * Get default workspace path
 *
 * Migrated from Python FastAPI: api/routes/files.py -> get_default_workspace
 */

import { NextResponse } from 'next/server';
import { fileService } from '@/lib/node-services/file-service';

export async function GET() {
  try {
    const result = await fileService.getDefaultWorkspace();
    return NextResponse.json(result);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);

    return NextResponse.json(
      { error: errorMessage, detail: errorMessage },
      { status: 500 }
    );
  }
}
