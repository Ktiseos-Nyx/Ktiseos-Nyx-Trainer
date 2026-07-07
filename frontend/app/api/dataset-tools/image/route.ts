import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import { createReadStream } from 'fs';
import path from 'path';
import mime from 'mime-types';
import { assertWithinBase } from '@/lib/dataset-tools/base-path';

const IMAGE_EXTENSIONS = new Set([
  '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif',
  '.svg', '.ico', '.avif', '.heic', '.heif', '.jxl',
]);

function isAllowedImage(filePath: string): boolean {
  return IMAGE_EXTENSIONS.has(path.extname(filePath).toLowerCase());
}

function isInternalRequest(request: Request): boolean {
  if (process.env.NODE_ENV !== 'production') return true;
  return request.headers.get('x-internal-request') === 'true';
}

export async function GET(request: Request) {
  if (!isInternalRequest(request)) {
    return NextResponse.json({ error: 'Access denied' }, { status: 403 });
  }

  const { searchParams } = new URL(request.url);
  const filePath = searchParams.get('path');
  const baseFolder = searchParams.get('baseFolder') || '.';

  if (!filePath) {
    return NextResponse.json({ error: 'File path is required' }, { status: 400 });
  }

  // Confine to the trainer's project root.
  let resolvedPath: string;
  try {
    const target = path.isAbsolute(filePath) ? filePath : path.join(/*turbopackIgnore: true*/ baseFolder, filePath);
    resolvedPath = assertWithinBase(target);
  } catch {
    return NextResponse.json({ error: 'Access denied - path outside project root' }, { status: 403 });
  }

  // Extension allowlist — reject non-image files
  if (!isAllowedImage(resolvedPath)) {
    return NextResponse.json({ error: 'File type not allowed' }, { status: 415 });
  }

  try {
    const stat = await fs.stat(resolvedPath);
    const etag = `"${stat.mtimeMs}-${stat.size}"`;
    const lastModified = new Date(stat.mtimeMs).toUTCString();

    // Return 304 if the client already has the current version
    const ifNoneMatch = request.headers.get('if-none-match');
    const ifModifiedSince = request.headers.get('if-modified-since');
    if (
      (ifNoneMatch && ifNoneMatch === etag) ||
      (!ifNoneMatch && ifModifiedSince && new Date(ifModifiedSince) >= new Date(stat.mtimeMs))
    ) {
      return new NextResponse(null, {
        status: 304,
        headers: { 'ETag': etag, 'Last-Modified': lastModified, 'Cache-Control': 'private, max-age=3600' },
      });
    }

    const contentType = mime.lookup(resolvedPath) || 'application/octet-stream';

    // Stream the file instead of loading it entirely into memory
    const nodeStream = createReadStream(resolvedPath);
    const webStream = new ReadableStream({
      start(controller) {
        nodeStream.on('data', (chunk) => controller.enqueue(chunk));
        nodeStream.on('end', () => controller.close());
        nodeStream.on('error', (err) => controller.error(err));
      },
    });

    return new NextResponse(webStream, {
      headers: {
        'Content-Type': contentType,
        'ETag': etag,
        'Last-Modified': lastModified,
        'Cache-Control': 'private, max-age=3600',
        'Content-Length': String(stat.size),
      },
    });
  } catch (error) {
    if (error instanceof Error) {
      if ('code' in error && error.code === 'ENOENT') {
        return NextResponse.json({ error: 'File not found' }, { status: 404 });
      }
      return NextResponse.json({ error: error.message }, { status: 500 });
    }
    return NextResponse.json({ error: 'An unknown error occurred' }, { status: 500 });
  }
}
