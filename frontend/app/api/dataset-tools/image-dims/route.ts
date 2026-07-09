import { NextResponse } from 'next/server';
import sharp from 'sharp';
import path from 'path';
import { assertWithinBase } from '@/lib/dataset-tools/base-path';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const filePath = searchParams.get('path');
  const baseFolder = searchParams.get('baseFolder') || '.';

  if (!filePath) {
    return NextResponse.json({ error: 'path is required' }, { status: 400 });
  }

  let resolvedPath: string;
  try {
    const target = path.isAbsolute(filePath) ? filePath : path.join(/*turbopackIgnore: true*/ baseFolder, filePath);
    resolvedPath = assertWithinBase(target);
  } catch {
    return NextResponse.json({ error: 'Access denied - path outside project root' }, { status: 403 });
  }

  try {
    const metadata = await sharp(resolvedPath).metadata();
    return NextResponse.json({
      width: metadata.width ?? 0,
      height: metadata.height ?? 0,
    });
  } catch (error) {
    console.error('Image dimensions error:', error);
    return NextResponse.json({ error: 'Failed to read image dimensions' }, { status: 500 });
  }
}
