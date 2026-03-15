/**
 * Shared dataset directory resolution and image extension constants.
 * Used by all dataset-related API routes.
 */
import fsSync from 'fs';
import path from 'path';

/** Image extensions recognized across all dataset routes. */
export const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'];

/**
 * Resolve the datasets root directory by checking an ordered set of locations.
 * Returns the first candidate path that actually exists on disk, or null if none do.
 */
export function getDatasetsDir(): string | null {
  const projectRoot = path.resolve(process.cwd(), '..');
  const candidates = [
    process.env.DATASETS_DIR,
    '/workspace/Ktiseos-Nyx-Trainer/datasets',
    path.join(process.cwd(), 'datasets'),
    path.join(projectRoot, 'datasets'),
    // Also check singular 'dataset/' for backwards compatibility
    path.join(process.cwd(), 'dataset'),
    path.join(projectRoot, 'dataset'),
  ].filter(Boolean) as string[];

  for (const dir of candidates) {
    if (fsSync.existsSync(dir)) return dir;
  }
  return null;
}

/**
 * Get datasets dir, creating it if it doesn't exist.
 * Used by routes that need to write (upload, etc).
 */
export function getOrCreateDatasetsDir(): string {
  const existing = getDatasetsDir();
  if (existing) return existing;

  // Fall back to cwd/datasets and let the caller create it
  return path.join(process.cwd(), 'datasets');
}
