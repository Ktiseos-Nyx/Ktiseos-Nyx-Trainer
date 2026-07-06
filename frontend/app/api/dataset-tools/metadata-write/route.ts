import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import { readPngParameters, writePngParameters, isPng } from '@/lib/dataset-tools/png-metadata';
import { assertWithinBase, resolveBase } from '@/lib/dataset-tools/base-path';

type Resolved = { path: string } | { error: string; status: number };

function isInternalRequest(request: Request): boolean {
  if (process.env.NODE_ENV !== 'production') return true;
  return request.headers.get('x-internal-request') === 'true';
}
async function resolveTarget(filePath: string, baseFolder: string): Promise<Resolved> {
  let resolvedPath: string;
  try {
    const target = path.isAbsolute(filePath) ? filePath : path.join(baseFolder || '.', filePath);
    resolvedPath = assertWithinBase(target);
  } catch {
    return { error: 'Access denied - path outside project root', status: 403 };
  }

  if (!resolvedPath.toLowerCase().endsWith('.png')) {
    return { error: 'Only PNG files are supported', status: 415 };
  }

  return { path: resolvedPath };
}

// GET /api/metadata-write?path=&baseFolder=
// Returns the raw 'parameters' tEXt string to prefill the editor: { text: string | null }.
export async function GET(request: Request) {
  if (!isInternalRequest(request)) {
    return NextResponse.json({ error: 'Access denied - internal requests only' }, { status: 403 });
  }

  const { searchParams } = new URL(request.url);
  const filePath = searchParams.get('path');
  const baseFolder = searchParams.get('baseFolder') || '.';

  if (!filePath) {
    return NextResponse.json({ error: 'File path is required' }, { status: 400 });
  }
  const resolved = await resolveTarget(filePath, baseFolder);
  if ('error' in resolved) {
    return NextResponse.json({ error: resolved.error }, { status: resolved.status });
  }

  try {
    const buf = await fs.readFile(resolved.path);
    if (!isPng(buf)) {
      return NextResponse.json({ error: 'Not a valid PNG file' }, { status: 415 });
    }
    return NextResponse.json({ text: readPngParameters(buf) });
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') {
      return NextResponse.json({ error: 'File not found' }, { status: 404 });
    }
    const message = error instanceof Error ? error.message : 'An unknown error occurred';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

// POST /api/metadata-write  { path, baseFolder, text, saveAsCopy? }
// Swaps the 'parameters' tEXt chunk in place (or writes name_edited.png) without
// recompressing pixels. Returns { ok: true, path: <basename written> }.
export async function POST(request: Request) {
  if (!isInternalRequest(request)) {
    return NextResponse.json({ error: 'Access denied - internal requests only' }, { status: 403 });
  }

  let body: { path?: string; baseFolder?: string; text?: string; saveAsCopy?: boolean };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  const { path: filePath, baseFolder = '.', text, saveAsCopy = false } = body;
  if (!filePath || typeof text !== 'string') {
    return NextResponse.json({ error: 'path and text are required' }, { status: 400 });
  }

  const resolved = await resolveTarget(filePath, baseFolder);
  if ('error' in resolved) {
    return NextResponse.json({ error: resolved.error }, { status: resolved.status });
  }
  const sourcePath = resolved.path;

  try {
    const stat = await fs.stat(sourcePath);
    if (!stat.isFile()) {
      return NextResponse.json({ error: 'Target is not a file' }, { status: 400 });
    }

    const buf = await fs.readFile(sourcePath);
    // Throws on a non-PNG, so the only bytes this route ever writes are a valid
    // PNG derived from an existing PNG at this location.
    const out = writePngParameters(buf, text);

    // Build target path and re-validate it against the base to prevent a
    // symlink redirect on the sibling name.
    let targetPath = sourcePath;
    if (saveAsCopy) {
      const ext = path.extname(sourcePath);
      const stem = path.basename(sourcePath, ext);
      // Underscore, not a second dot: `foo_edited.png` keeps a single-extension
      // stem so dataset/training tools that pair by stem (foo.png ↔ foo.txt) or
      // split on the first dot don't orphan the file.
      targetPath = path.join(path.dirname(sourcePath), `${stem}_edited${ext}`);
    }

    // Re-validate the final write target against the project root, so a
    // symlink on the sibling name can't redirect the write outside the
    // confined tree.
    const writeResolved = await resolveTarget(targetPath, resolveBase());
    if ('error' in writeResolved) {
      return NextResponse.json({ error: writeResolved.error }, { status: writeResolved.status });
    }

    await fs.writeFile(writeResolved.path, out);
    return NextResponse.json({ ok: true, path: path.basename(writeResolved.path) });
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') {
      return NextResponse.json({ error: 'File not found' }, { status: 404 });
    }
    const message = error instanceof Error ? error.message : 'An unknown error occurred';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
