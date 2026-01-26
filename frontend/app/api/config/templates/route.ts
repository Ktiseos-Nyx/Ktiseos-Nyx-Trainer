/**
 * GET /api/config/templates
 * List available configuration templates
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import toml from '@iarna/toml';

export async function GET(request: NextRequest) {
  try {
    const projectRoot = path.resolve(process.cwd(), '..');
    const templatesDir = path.join(projectRoot, 'example_configs');

    try {
      await fs.access(templatesDir);
    } catch {
      return NextResponse.json({ templates: [] });
    }

    const files = await fs.readdir(templatesDir);
    const tomlFiles = files.filter(f => f.endsWith('.toml'));

    const templates = [];

    for (const file of tomlFiles) {
      try {
        const filePath = path.join(templatesDir, file);
        const content = await fs.readFile(filePath, 'utf-8');
        const configData = toml.parse(content) as any;

        templates.push({
          name: path.parse(file).name,
          path: filePath,
          description: configData.description || 'No description',
        });
      } catch (error) {
        console.warn(`Failed to load template ${file}:`, error);
      }
    }

    return NextResponse.json({ templates });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
