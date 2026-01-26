/**
 * POST /api/config/save
 * Save a configuration file
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import toml from '@iarna/toml';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { name, config } = body;

    if (!name || !config) {
      return NextResponse.json(
        { error: 'Missing required fields: name, config' },
        { status: 400 }
      );
    }

    const projectRoot = path.resolve(process.cwd(), '..');
    const configDir = path.join(projectRoot, 'training_config');

    // Ensure directory exists
    await fs.mkdir(configDir, { recursive: true });

    const filePath = path.join(configDir, `${name}.toml`);
    const tomlContent = toml.stringify(config as any);

    await fs.writeFile(filePath, tomlContent, 'utf-8');

    return NextResponse.json({
      success: true,
      path: filePath,
      message: `Configuration saved: ${name}`,
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
