/**
 * GET  /api/config/form  — Load last training form state (JSON)
 * POST /api/config/form  — Save current training form state (JSON)
 *
 * Provides server-side persistence for the training form so state survives
 * browser sessions where localStorage is cleared (adblockers, tracker
 * blockers, privacy browsers, fresh VastAI connections, etc.).
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

function formConfigPath(): string {
  const projectRoot = path.resolve(process.cwd(), '..');
  return path.join(projectRoot, 'config', 'last_training_form.json');
}

/**
 * GET /api/config/form
 * Returns the last saved training form state, or 404 if none exists yet.
 */
export async function GET() {
  try {
    const filePath = formConfigPath();
    const content = await fs.readFile(filePath, 'utf-8');
    return NextResponse.json({ success: true, config: JSON.parse(content) });
  } catch {
    return NextResponse.json({ success: false }, { status: 404 });
  }
}

/**
 * POST /api/config/form
 * Saves the current training form state for cross-session recovery.
 *
 * @param request - Body should be the full TrainingConfig object.
 */
export async function POST(request: NextRequest) {
  try {
    const config = await request.json();
    const filePath = formConfigPath();
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, JSON.stringify(config, null, 2), 'utf-8');
    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
