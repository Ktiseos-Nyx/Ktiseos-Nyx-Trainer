/**
 * GET /api/config/load?path=...
 * Load a configuration file
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import toml from '@iarna/toml';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const configPath = searchParams.get('path');

    if (!configPath) {
      return NextResponse.json(
        { error: 'Missing required parameter: path' },
        { status: 400 }
      );
    }

    const filePath = path.resolve(configPath);

    try {
      await fs.access(filePath);
    } catch {
      return NextResponse.json(
        { error: `Configuration file not found: ${configPath}` },
        { status: 404 }
      );
    }

    const content = await fs.readFile(filePath, 'utf-8');
    const config = toml.parse(content);

    return NextResponse.json({
      success: true,
      config,
      path: filePath,
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
