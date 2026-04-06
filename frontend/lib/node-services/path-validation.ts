/**
 * Shared path validation utility for security
 * Prevents path traversal attacks by confining operations to allowed directories.
 *
 * Extracted from file-service.ts patterns, used across all API routes.
 */

import fs from 'fs';
import path from 'path';

// Project root: process.cwd() is frontend/, go up one level
const PROJECT_ROOT = path.resolve(process.cwd(), '..');

// Specific allowed directories for different operation types
// Support both 'datasets' (plural) and 'dataset' (singular) directories
const DATASETS_DIR = path.join(PROJECT_ROOT, 'datasets');
const DATASET_DIR_ALT = path.join(PROJECT_ROOT, 'dataset');
const OUTPUT_DIR = path.join(PROJECT_ROOT, 'output');
const MODELS_DIR = path.join(PROJECT_ROOT, 'pretrained_model');
const VAE_DIR = path.join(PROJECT_ROOT, 'vae');
const CONFIG_DIR = path.join(PROJECT_ROOT, 'config');
const PRESETS_DIR = path.join(PROJECT_ROOT, 'presets');
const RUNTIME_STORE_DIR = path.join(PROJECT_ROOT, 'trainer', 'runtime_store');

// General browsing includes project root + home dir (for file manager)
const BROWSE_DIRS = [
  PROJECT_ROOT,
  process.env.HOME || process.env.USERPROFILE || '',
].filter(Boolean);

/**
 * Canonicalize a path by resolving symlinks for the existing portion
 * and appending any not-yet-created tail segments.
 */
function canonicalizeForCheck(inputPath: string): string {
  const resolved = path.resolve(inputPath);
  let cursor = resolved;
  const suffix: string[] = [];

  while (!fs.existsSync(cursor)) {
    const parent = path.dirname(cursor);
    if (parent === cursor) break;
    suffix.unshift(path.basename(cursor));
    cursor = parent;
  }

  const realBase = fs.existsSync(cursor)
    ? fs.realpathSync.native(cursor)
    : path.resolve(cursor);

  return suffix.length ? path.join(realBase, ...suffix) : realBase;
}

/**
 * Check if a resolved path is within any of the given allowed directories.
 */
function isWithin(resolvedPath: string, allowedDirs: string[]): boolean {
  const normalized = canonicalizeForCheck(resolvedPath);
  return allowedDirs.some(dir => {
    const normalizedDir = canonicalizeForCheck(dir);
    return normalized === normalizedDir ||
      normalized.startsWith(normalizedDir + path.sep);
  });
}

/**
 * Validate that a path is within allowed directories.
 * Resolves the path, canonicalizes symlinks, and checks confinement.
 *
 * @throws Error if path is outside allowed directories
 */
function validateWithin(userPath: string, allowedDirs: string[], context: string): string {
  const resolved = canonicalizeForCheck(userPath);
  if (!isWithin(resolved, allowedDirs)) {
    throw new Error(`Access denied: ${context} path outside allowed directories`);
  }
  return resolved;
}

// ========== Public API ==========

/** Validate a dataset path (must be under datasets/ or dataset/) */
export function validateDatasetPath(userPath: string): string {
  return validateWithin(userPath, [DATASETS_DIR, DATASET_DIR_ALT], 'Dataset');
}

/** Validate an image path (must be under datasets/ or dataset/) */
export function validateImagePath(userPath: string): string {
  return validateWithin(userPath, [DATASETS_DIR, DATASET_DIR_ALT], 'Image');
}

/** Validate a config file path (must be under config/, presets/, or runtime_store/) */
export function validateConfigPath(userPath: string): string {
  return validateWithin(userPath, [CONFIG_DIR, PRESETS_DIR, RUNTIME_STORE_DIR], 'Config');
}

/** Validate an output path (must be under output/) */
export function validateOutputPath(userPath: string): string {
  return validateWithin(userPath, [OUTPUT_DIR], 'Output');
}

/** Validate a model path (must be under pretrained_model/ or vae/) */
export function validateModelPath(userPath: string): string {
  return validateWithin(userPath, [MODELS_DIR, VAE_DIR], 'Model');
}

/** Validate a general browsing path (project root or home - for file manager) */
export function validateBrowsePath(userPath: string): string {
  return validateWithin(userPath, BROWSE_DIRS, 'Browse');
}

/** Generic: validate path against arbitrary allowed directories */
export function validatePathWithin(userPath: string, allowedDirs: string[], context = 'Path'): string {
  return validateWithin(userPath, allowedDirs, context);
}

export { PROJECT_ROOT, DATASETS_DIR, OUTPUT_DIR, MODELS_DIR, VAE_DIR, CONFIG_DIR };
