/**
 * Next.js API Route: DELETE /api/settings/user/[key]
 * Delete a specific setting key
 *
 * Migrated from Python FastAPI: api/routes/settings.py -> delete_setting_key
 */

import { NextRequest, NextResponse } from 'next/server';
import { settingsService } from '@/lib/node-services/settings-service';

export async function DELETE(
  request: NextRequest,
  { params }: { params: { key: string } }
) {
  try {
    const { key } = params;

    // Validate key
    const validKeys = ['huggingface_token', 'civitai_api_key'];
    if (!validKeys.includes(key)) {
      return NextResponse.json(
        { error: `Invalid key. Must be one of: ${validKeys.join(', ')}` },
        { status: 400 }
      );
    }

    const result = await settingsService.deleteSettingKey(
      key as 'huggingface_token' | 'civitai_api_key'
    );

    if (!result.success) {
      return NextResponse.json(
        { error: result.error || 'Failed to delete setting' },
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
