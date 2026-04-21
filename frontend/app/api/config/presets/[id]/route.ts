/**
 * Preset by ID Routes
 * GET - Get specific preset
 * DELETE - Delete preset
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import { validateConfigPath } from '@/lib/node-services/path-validation';

/**
 * Sanitize preset ID to prevent path traversal.
 * Only allow alphanumeric, hyphens, underscores.
 */
function sanitizePresetId(id: string): string {
  // Block traversal and directory separator characters; allow spaces/dots that appear in preset filenames
  if (/[/\\]/.test(id) || id.includes('..')) {
    throw new Error('Invalid preset ID');
  }
  const trimmed = id.trim();
  if (!trimmed) {
    throw new Error('Invalid preset ID');
  }
  return trimmed;
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    if (!id) {
      return NextResponse.json(
        { error: 'Missing preset ID' },
        { status: 400 }
      );
    }

    // Security: sanitize ID and validate resulting path
    let filePath: string;
    try {
      const safeId = sanitizePresetId(id);
      const projectRoot = path.resolve(process.cwd(), '..');
      const candidatePath = path.join(projectRoot, 'presets', `${safeId}.json`);
      filePath = validateConfigPath(candidatePath);
    } catch {
      return NextResponse.json(
        { error: 'Access denied: invalid preset ID' },
        { status: 403 }
      );
    }

    try {
      await fs.access(filePath);
    } catch {
      return NextResponse.json(
        { error: `Preset not found: ${id}` },
        { status: 404 }
      );
    }

    const content = await fs.readFile(filePath, 'utf-8');
    const preset = JSON.parse(content);

    // Old community presets (bmaltais GUI format) nest all training keys under a
    // "config" sub-object with some different key names (e.g. LoRA_type).  Hoist
    // them to the root so the frontend TrainingConfig fields are populated correctly.
    let responseData: Record<string, unknown> = { id, ...preset };
    if (preset.config && typeof preset.config === 'object' && !Array.isArray(preset.config)) {
      const { config: nestedConfig, ...rest } = preset;
      const { LoRA_type, ...otherConfig } = nestedConfig as Record<string, unknown>;
      responseData = {
        id,
        ...rest,
        ...(LoRA_type !== undefined ? { lora_type: LoRA_type } : {}),
        ...otherConfig,
      };
    }

    return NextResponse.json(responseData);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    if (!id) {
      return NextResponse.json(
        { error: 'Missing preset ID' },
        { status: 400 }
      );
    }

    // Security: sanitize ID and validate resulting path
    let filePath: string;
    try {
      const safeId = sanitizePresetId(id);
      const projectRoot = path.resolve(process.cwd(), '..');
      const candidatePath = path.join(projectRoot, 'presets', `${safeId}.json`);
      filePath = validateConfigPath(candidatePath);
    } catch {
      return NextResponse.json(
        { error: 'Access denied: invalid preset ID' },
        { status: 403 }
      );
    }

    try {
      await fs.access(filePath);
    } catch {
      return NextResponse.json(
        { error: `Preset not found: ${id}` },
        { status: 404 }
      );
    }

    await fs.unlink(filePath);

    return NextResponse.json({
      success: true,
      message: `Preset deleted: ${id}`,
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
