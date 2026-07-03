/**
 * fs-base resolver for the ported Dataset-Tools routes.
 *
 * All `/api/dataset-tools/*` routes read (and a couple write) files by path
 * supplied from the browser. `resolveBase()` anchors to the trainer's
 * repo root and `assertWithinBase()` is the single confinement gate every
 * route calls before touching the filesystem — it throws on any attempt to
 * escape that root (`..` traversal, an absolute path outside the root, or a
 * symlink that resolves outside the root).
 */

import fs from 'fs';
import path from 'path';

/**
 * Walk up from `startDir` looking for the trainer repo root, identified by
 * `AGENTS.md` sitting next to a `frontend/` directory. This is file-anchored
 * rather than a fixed `path.resolve(process.cwd(), '..')` hop, so it resolves
 * correctly no matter what directory the Next.js server process was started
 * from (`frontend/` in normal dev/prod, but potentially the repo root or
 * elsewhere depending on the launcher).
 */
function findProjectRoot(startDir: string): string {
  let dir = path.resolve(startDir);

  while (true) {
    if (fs.existsSync(path.join(dir, 'AGENTS.md')) && fs.existsSync(path.join(dir, 'frontend'))) {
      return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) {
      throw new Error(
        `Could not locate the Ktiseos-Nyx-Trainer project root (no AGENTS.md + frontend/ found walking up from ${startDir})`,
      );
    }
    dir = parent;
  }
}

const PROJECT_ROOT = findProjectRoot(process.cwd());

/**
 * Resolve symlinks for the existing portion of `inputPath` and re-append any
 * not-yet-created tail segments, so a symlink can't be used to redirect a
 * confinement check to a location outside the intended root — including for
 * paths (e.g. a new "_edited.png" sibling) that don't exist yet.
 *
 * Mirrors the canonicalization approach already used by
 * `frontend/lib/node-services/path-validation.ts`.
 */
function canonicalize(inputPath: string): string {
  const resolved = path.resolve(inputPath);
  let cursor = resolved;
  const tail: string[] = [];

  while (!fs.existsSync(cursor)) {
    const parent = path.dirname(cursor);
    if (parent === cursor) break;
    tail.unshift(path.basename(cursor));
    cursor = parent;
  }

  const realBase = fs.existsSync(cursor) ? fs.realpathSync.native(cursor) : path.resolve(cursor);
  return tail.length ? path.join(realBase, ...tail) : realBase;
}

/** The trainer's project root — file-anchored, independent of process.cwd(). */
export function resolveBase(): string {
  return PROJECT_ROOT;
}

/**
 * Resolve `target` against the project root and verify the result does not
 * escape it. Accepts an absolute or relative path; rejects `..` traversal,
 * absolute paths outside the root, and symlinks that resolve outside it.
 *
 * @throws Error if `target` resolves outside the project root.
 * @returns the safe, symlink-resolved absolute path.
 */
export function assertWithinBase(target: string): string {
  const base = canonicalize(resolveBase());
  const resolved = canonicalize(path.resolve(base, target));

  if (resolved !== base && !resolved.startsWith(base + path.sep)) {
    throw new Error(`Path traversal blocked: "${target}" resolves outside the project root`);
  }

  return resolved;
}
