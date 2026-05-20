/**
 * Next.js API Route: POST /api/dataset/upload-chunk-finalize
 * Assembles all uploaded chunks and extracts the ZIP into a dataset.
 *
 * Reads the session's part_XXXX files in index order, concatenates them
 * into a single Buffer, then proxies that buffer to the FastAPI
 * /api/dataset/upload-zip endpoint which handles the ZIP extraction.
 * Temp chunks are always cleaned up, even on failure.
 */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import { createWriteStream } from 'fs';
import os from 'os';
import path from 'path';

/** Resolve the FastAPI backend base URL — mirrors the pattern used in upload-zip/route.ts. */
function backendBase(): string {
  const dev = process.env.NODE_ENV !== 'production';
  const defaultPort = dev ? '8000' : (process.env.BACKEND_PORT || '18000');
  return process.env.BACKEND_URL || `http://127.0.0.1:${defaultPort}`;
}

interface SessionMeta {
  totalChunks: number;
  fileName: string;
}

/**
 * Assemble all chunks for an upload session and extract the resulting ZIP.
 *
 * Expects JSON body `{ uploadId, fileName, datasetName }`.
 * Concatenates `part_0000` … `part_NNNN` in order, posts the assembled
 * buffer to the FastAPI upload-zip endpoint, then deletes the scratch dir.
 *
 * @returns The FastAPI extraction response on success.
 */
export async function POST(request: NextRequest) {
  let chunkDir: string | null = null;

  try {
    const body = await request.json() as { uploadId?: unknown; fileName?: unknown; datasetName?: unknown };
    const { uploadId, fileName, datasetName } = body;

    if (typeof uploadId !== 'string' || typeof fileName !== 'string' || typeof datasetName !== 'string') {
      return NextResponse.json(
        { error: 'uploadId, fileName, and datasetName are required strings' },
        { status: 400 },
      );
    }

    const safeId = uploadId.replace(/[^a-zA-Z0-9-]/g, '');
    chunkDir = path.join(os.tmpdir(), 'ktiseos_uploads', safeId);

    const metaRaw = await fs.readFile(path.join(chunkDir, 'meta.json'), 'utf-8');
    const meta = JSON.parse(metaRaw) as SessionMeta;

    // Validate all chunks are present before attempting assembly
    for (let i = 0; i < meta.totalChunks; i++) {
      const chunkPath = path.join(chunkDir, `part_${String(i).padStart(4, '0')}`);
      try {
        await fs.access(chunkPath);
      } catch {
        return NextResponse.json(
          { error: `Missing chunk ${i} of ${meta.totalChunks} — upload may be incomplete` },
          { status: 400 },
        );
      }
    }

    // Stream chunks into a single temp file (avoids loading all chunks into RAM simultaneously)
    const assembledPath = path.join(chunkDir, 'assembled.zip');
    await new Promise<void>((resolve, reject) => {
      const ws = createWriteStream(assembledPath);
      ws.on('error', reject);
      ws.on('finish', resolve);
      (async () => {
        for (let i = 0; i < meta.totalChunks; i++) {
          const chunkPath = path.join(chunkDir, `part_${String(i).padStart(4, '0')}`);
          const data = await fs.readFile(chunkPath);
          if (!ws.write(data)) await new Promise<void>(r => ws.once('drain', r));
        }
        ws.end();
      })().catch(reject);
    });

    // Read assembled file once for the FormData body
    const assembled = await fs.readFile(assembledPath);

    // Proxy assembled file to FastAPI for ZIP extraction
    const formData = new FormData();
    formData.append('file', new Blob([assembled], { type: 'application/zip' }), fileName);
    formData.append('dataset_name', datasetName);

    const res = await fetch(`${backendBase()}/api/dataset/upload-zip`, {
      method: 'POST',
      body: formData,
    });

    const contentType = res.headers.get('content-type') || '';

    if (!res.ok) {
      const errText = await res.text();
      return NextResponse.json(
        { error: `Backend ${res.status}`, detail: errText.slice(0, 1000) },
        { status: res.status },
      );
    }

    if (contentType.includes('application/json')) {
      return NextResponse.json(await res.json(), { status: res.status });
    }

    const text = await res.text();
    try {
      return NextResponse.json(JSON.parse(text), { status: res.status });
    } catch {
      return NextResponse.json({ success: true, detail: text.slice(0, 1000) }, { status: res.status });
    }
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ error: 'Finalize failed', detail: msg }, { status: 500 });
  } finally {
    // Always clean up — even if FastAPI returned an error
    if (chunkDir) {
      fs.rm(chunkDir, { recursive: true, force: true }).catch(() => {});
    }
  }
}
