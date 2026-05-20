/**
 * Next.js API Route: POST /api/dataset/upload-chunk
 * Receives one chunk of a chunked ZIP upload and writes it to disk.
 *
 * Expects multipart/form-data with:
 *   chunk   – binary file slice
 *   uploadId – session ID from /upload-chunk-init
 *   index   – zero-based chunk index (used for ordered assembly)
 */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';

/**
 * Write one chunk to the upload session's scratch directory.
 *
 * Chunks are stored as `part_XXXX` (zero-padded index) so the finalize
 * route can reassemble them in order without metadata lookups.
 *
 * @returns `{ status: 'chunk_saved', index }` on success.
 */
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const chunk = formData.get('chunk');
    const uploadId = formData.get('uploadId');
    const rawIndex = formData.get('index');

    if (!(chunk instanceof File) || typeof uploadId !== 'string' || typeof rawIndex !== 'string') {
      return NextResponse.json({ error: 'chunk (file), uploadId, and index are required' }, { status: 400 });
    }

    const index = parseInt(rawIndex, 10);
    if (isNaN(index) || index < 0) {
      return NextResponse.json({ error: 'index must be a non-negative integer' }, { status: 400 });
    }

    const safeId = uploadId.replace(/[^a-zA-Z0-9-]/g, '');
    const chunkDir = path.join(os.tmpdir(), 'ktiseos_uploads', safeId);

    // Reject chunks for sessions that were never initialised
    try {
      await fs.access(chunkDir);
    } catch {
      return NextResponse.json(
        { error: 'Upload session not found — call /upload-chunk-init first' },
        { status: 404 },
      );
    }

    const chunkPath = path.join(chunkDir, `part_${String(index).padStart(4, '0')}`);
    await fs.writeFile(chunkPath, Buffer.from(await chunk.arrayBuffer()));

    return NextResponse.json({ status: 'chunk_saved', index });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
