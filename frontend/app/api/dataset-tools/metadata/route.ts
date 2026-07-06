import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import mime from 'mime-types';
import { assertWithinBase } from '@/lib/dataset-tools/base-path';
import { extractMetadataFromBuffer } from '@/lib/dataset-tools/metadata-extract';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const filePath = searchParams.get('path');
  const baseFolder = searchParams.get('baseFolder') || '.';

  if (!filePath) {
    return NextResponse.json({ error: 'File path is required' }, { status: 400 });
  }

  let resolvedPath: string;
  try {
    const target = path.isAbsolute(filePath) ? filePath : path.join(baseFolder, filePath);
    resolvedPath = assertWithinBase(target);
  } catch {
    return NextResponse.json({ error: 'Access denied - path outside project root' }, { status: 403 });
  }

  try {
    const file = await fs.readFile(resolvedPath);
    const stats = await fs.stat(resolvedPath);
    const mimeType = mime.lookup(resolvedPath) || 'application/octet-stream';

    const metadata = await extractMetadataFromBuffer(
      file,
      mimeType,
      path.basename(resolvedPath),
      stats.size,
      stats.mtime.toISOString(),
    );

    return NextResponse.json(metadata);
  } catch (error) {
    console.error('Metadata extraction error:', error);
    if (error instanceof Error) {
      if ('code' in error && error.code === 'ENOENT') {
        return NextResponse.json({ error: 'File not found' }, { status: 404 });
      }
      return NextResponse.json({ error: error.message }, { status: 500 });
    }
    return NextResponse.json({ error: 'An unknown error occurred' }, { status: 500 });
  }
}
