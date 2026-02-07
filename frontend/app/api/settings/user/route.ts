/**
 * Next.js API Routes: /api/settings/user
 * User settings management
 *
 * Migrated from Python FastAPI: api/routes/settings.py
 */

import { NextRequest, NextResponse } from 'next/server';
import { settingsService } from '@/lib/node-services/settings-service';

/**
 * GET /api/settings/user
 * Get user settings with masked API keys
 */
export async function GET() {
  try {
    const result = await settingsService.getUserSettings();

    if (!result.success) {
      return NextResponse.json(
        { error: result.error || 'Failed to get settings' },
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

/**
 * POST /api/settings/user
 * Update user settings
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const result = await settingsService.updateUserSettings({
      huggingface_token: body.huggingface_token,
      civitai_api_key: body.civitai_api_key,
    });

    if (!result.success) {
      return NextResponse.json(
        { error: result.error || 'Failed to update settings' },
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

/**
 * DELETE /api/settings/user
 * Clear all user settings
 */
export async function DELETE() {
  try {
    const result = await settingsService.clearUserSettings();

    if (!result.success) {
      return NextResponse.json(
        { error: result.error || 'Failed to clear settings' },
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
