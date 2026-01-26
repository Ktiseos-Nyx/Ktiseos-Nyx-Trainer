/**
 * Preset by ID Routes
 * GET - Get specific preset
 * DELETE - Delete preset
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params;

    if (!id) {
      return NextResponse.json(
        { error: 'Missing preset ID' },
        { status: 400 }
      );
    }

    const projectRoot = path.resolve(process.cwd(), '..');
    const filePath = path.join(projectRoot, 'presets', `${id}.json`);

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
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params;

    if (!id) {
      return NextResponse.json(
        { error: 'Missing preset ID' },
        { status: 400 }
      );
    }

    const projectRoot = path.resolve(process.cwd(), '..');
    const filePath = path.join(projectRoot, 'presets', `${id}.json`);

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
