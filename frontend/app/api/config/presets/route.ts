/**
 * Presets API Routes
 * GET - List all presets
 * POST - Create new preset
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

export async function GET(request: NextRequest) {
  try {
    const projectRoot = path.resolve(process.cwd(), '..');
    const presetsDir = path.join(projectRoot, 'presets');

    try {
      await fs.access(presetsDir);
    } catch {
      return NextResponse.json({ presets: [] });
    }

    const files = await fs.readdir(presetsDir);
    const jsonFiles = files.filter(f => f.endsWith('.json'));

    const presets = [];

    for (const file of jsonFiles) {
      try {
        const filePath = path.join(presetsDir, file);
        const content = await fs.readFile(filePath, 'utf-8');
        const preset = JSON.parse(content);

        presets.push({
          id: path.parse(file).name,
          name: preset.name || path.parse(file).name,
          description: preset.description || '',
          model_type: preset.model_type,
          config: preset,
        });
      } catch (error) {
        console.warn(`Failed to load preset ${file}:`, error);
      }
    }

    return NextResponse.json({ presets });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { name, description, model_type, config } = body;

    if (!name || !config) {
      return NextResponse.json(
        { error: 'Missing required fields: name, config' },
        { status: 400 }
      );
    }

    const projectRoot = path.resolve(process.cwd(), '..');
    const presetsDir = path.join(projectRoot, 'presets');

    // Ensure directory exists
    await fs.mkdir(presetsDir, { recursive: true });

    // Create preset object
    const preset = {
      name,
      description: description || '',
      model_type: model_type || null,
      created_at: Date.now(),
      ...config,
    };

    // Sanitize filename
    const filename = name.replace(/[^a-z0-9_-]/gi, '_').toLowerCase();
    const filePath = path.join(presetsDir, `${filename}.json`);

    await fs.writeFile(filePath, JSON.stringify(preset, null, 2), 'utf-8');

    return NextResponse.json({
      success: true,
      id: filename,
      message: `Preset saved: ${name}`,
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
