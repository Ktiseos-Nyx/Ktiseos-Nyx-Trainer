/**
 * Next.js API Routes: /api/settings/user/[key]
 * Get or delete a specific setting key
 *
 * Migrated from Python FastAPI: api/routes/settings.py -> delete_setting_key
 */

import { NextRequest, NextResponse } from 'next/server';
import { settingsService } from '@/lib/node-services/settings-service';

const VALID_KEYS = ['huggingface_token', 'civitai_api_key'] as const;
type ValidKey = typeof VALID_KEYS[number];

/**
 * GET /api/settings/user/[key]
 * Returns whether a stored API key is set — never the raw value.
 * Only valid for huggingface_token and civitai_api_key.
 */
export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ key: string }> }
) {
  try {
    const { key } = await params;

    if (!VALID_KEYS.includes(key as ValidKey)) {
      return NextResponse.json(
        { error: `Invalid key. Must be one of: ${VALID_KEYS.join(', ')}` },
        { status: 400 }
      );
    }

    const keys = await settingsService.getApiKeys();
    const isSet = Boolean(keys[key as ValidKey]);
    return NextResponse.json({ key, isSet });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ key: string }> }
) {
  try {
    const { key } = await params;

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
