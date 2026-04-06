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
  const sanitized = id.replace(/[^a-zA-Z0-9_-]/g, '');
  if (!sanitized) {
    throw new Error('Invalid preset ID');
  }
  return sanitized;
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

    return NextResponse.json({
      id,
      ...preset,
    });
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
