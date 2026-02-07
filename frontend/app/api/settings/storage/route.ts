/**
 * Next.js API Route: GET /api/settings/storage
 * Get storage information
 *
 * Migrated from Python FastAPI: api/routes/settings.py -> get_storage_info
 */

import { NextResponse } from 'next/server';
import { settingsService } from '@/lib/node-services/settings-service';

export async function GET() {
  try {
    const result = await settingsService.getStorageInfo();

    if (!result.success) {
      return NextResponse.json(
        { error: result.error || 'Failed to get storage info' },
        { status: 500 }
      );
    }

    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
