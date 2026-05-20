/**
 * Next.js API Route: POST /api/dataset/upload-chunk-init
 * Opens a chunked-upload session for a single ZIP file.
 *
 * The client calls this once before sending chunks, receiving an upload_id
 * it passes to every subsequent /upload-chunk and /upload-chunk-finalize call.
 * Chunks are stored under os.tmpdir()/ktiseos_uploads/<uploadId>/.
 */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';

/**
 * Initialise a chunked-upload session.
 *
 * Expects JSON body `{ uploadId, fileName, totalChunks }`.
 * Creates a scratch directory in the OS temp folder and writes session
 * metadata so the finalize route knows how many parts to assemble.
 *
 * @returns `{ status: 'ready', uploadId }` on success.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json() as { uploadId?: unknown; fileName?: unknown; totalChunks?: unknown };
    const { uploadId, fileName, totalChunks } = body;

    if (typeof uploadId !== 'string' || typeof fileName !== 'string' || typeof totalChunks !== 'number') {
      return NextResponse.json({ error: 'uploadId (string), fileName (string), and totalChunks (number) are required' }, { status: 400 });
    }

    // UUID-only characters — prevents path traversal
    const safeId = uploadId.replace(/[^a-zA-Z0-9-]/g, '');
    if (!safeId) {
      return NextResponse.json({ error: 'Invalid uploadId' }, { status: 400 });
    }

    const chunkDir = path.join(os.tmpdir(), 'ktiseos_uploads', safeId);
    await fs.mkdir(chunkDir, { recursive: true });
    await fs.writeFile(
      path.join(chunkDir, 'meta.json'),
      JSON.stringify({ uploadId: safeId, fileName, totalChunks, createdAt: Date.now() }),
    );

    return NextResponse.json({ status: 'ready', uploadId: safeId });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
