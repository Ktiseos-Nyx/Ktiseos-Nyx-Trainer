/**
 * File Service - Node.js file operations
 * Migrated from Python api/routes/files.py
 *
 * Handles:
 * - Directory listing with security checks
 * - File upload/download
 * - File deletion and renaming
 * - Directory creation
 * - Text file read/write
 */

import fs from 'fs/promises';
import path from 'path';
import { Stats } from 'fs';

// ========== Types ==========

export interface FileInfo {
  name: string;
  path: string;
  type: 'file' | 'dir';
  size: number;
  modified: number;
  is_image?: boolean;
  mime_type?: string;
}

export interface DirectoryListing {
  path: string;
  parent: string | null;
  files: FileInfo[];
}

export interface FileOperationResult {
  success: boolean;
  message?: string;
  path?: string;
  size?: number;
  error?: string;
}

// ========== Configuration ==========

// Project root detection
// process.cwd() returns the frontend directory in Next.js
// Go up one level to get to project root
const PROJECT_ROOT = path.join(process.cwd(), '..').replace(/\\/g, '/');
const DEFAULT_WORKSPACE = PROJECT_ROOT;

// Allowed directories for security
const ALLOWED_DIRS = [
  PROJECT_ROOT,
  process.env.HOME || process.env.USERPROFILE || ''
].filter(Boolean);

// Image MIME types
const IMAGE_MIME_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml'];

// ========== Security Functions ==========

/**
 * Check if a path is within allowed directories
 */
function isSafePath(userPath: string): boolean {
  try {
    const resolvedPath = path.resolve(userPath);
    return ALLOWED_DIRS.some(allowedDir => {
      const resolved = path.resolve(allowedDir);
      return resolvedPath.startsWith(resolved);
    });
  } catch {
    return false;
  }
}

/**
 * Validate and resolve a path safely
 */
function validatePath(userPath: string): string {
  const resolved = path.resolve(userPath);

  if (!isSafePath(resolved)) {
    throw new Error('Access denied: Path outside allowed directories');
  }

  return resolved;
}

// ========== Helper Functions ==========

/**
 * Get MIME type from file extension
 */
function getMimeType(filePath: string): string | null {
  const ext = path.extname(filePath).toLowerCase();
  const mimeTypes: Record<string, string> = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.svg': 'image/svg+xml',
    '.txt': 'text/plain',
    '.json': 'application/json',
    '.pdf': 'application/pdf',
    '.zip': 'application/zip',
  };

  return mimeTypes[ext] || null;
}

/**
 * Get file information from stats
 */
async function getFileInfo(filePath: string): Promise<FileInfo> {
  const stats: Stats = await fs.stat(filePath);
  const mimeType = getMimeType(filePath);

  return {
    name: path.basename(filePath),
    path: filePath,
    type: stats.isDirectory() ? 'dir' : 'file',
    size: stats.isFile() ? stats.size : 0,
    modified: stats.mtimeMs,
    is_image: mimeType ? IMAGE_MIME_TYPES.includes(mimeType) : false,
    mime_type: mimeType || undefined,
  };
}

// ========== FileService Class ==========

export class FileService {
  /**
   * Get the default workspace path
   */
  async getDefaultWorkspace(): Promise<{ path: string; allowed_dirs: string[] }> {
    return {
      path: DEFAULT_WORKSPACE,
      allowed_dirs: ALLOWED_DIRS,
    };
  }

  /**
   * List directory contents with security checks
   */
  async listDirectory(dirPath?: string): Promise<DirectoryListing> {
    try {
      // Use default workspace if no path provided
      const targetPath = dirPath ? validatePath(dirPath) : DEFAULT_WORKSPACE;

      // Check if path exists
      try {
        await fs.access(targetPath);
      } catch {
        throw new Error('Path not found');
      }

      // Check if it's a directory
      const stats = await fs.stat(targetPath);
      if (!stats.isDirectory()) {
        throw new Error('Path is not a directory');
      }

      // Get parent directory
      const parent = path.dirname(targetPath) !== targetPath ? path.dirname(targetPath) : null;

      // List contents
      const entries = await fs.readdir(targetPath);
      const files: FileInfo[] = [];

      for (const entry of entries) {
        try {
          const fullPath = path.join(targetPath, entry);
          const fileInfo = await getFileInfo(fullPath);
          files.push(fileInfo);
        } catch (error) {
          // Skip files we can't read
          console.warn(`Failed to get info for ${entry}:`, error);
        }
      }

      // Sort: directories first, then by name
      files.sort((a, b) => {
        if (a.type !== b.type) {
          return a.type === 'dir' ? -1 : 1;
        }
        return a.name.toLowerCase().localeCompare(b.name.toLowerCase());
      });

      return {
        path: targetPath,
        parent,
        files,
      };
    } catch (error) {
      throw new Error(`Failed to list directory: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Read a text file
   */
  async readFile(filePath: string): Promise<{ path: string; content: string; size: number }> {
    try {
      const validPath = validatePath(filePath);

      // Check if file exists
      try {
        await fs.access(validPath);
      } catch {
        throw new Error('File not found');
      }

      // Check if it's a file
      const stats = await fs.stat(validPath);
      if (!stats.isFile()) {
        throw new Error('Not a file');
      }

      // Read file content
      const content = await fs.readFile(validPath, 'utf-8');

      return {
        path: validPath,
        content,
        size: stats.size,
      };
    } catch (error) {
      if (error instanceof Error && error.message.includes('ENOENT')) {
        throw new Error('File not found');
      }
      throw new Error(`Failed to read file: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Write a text file
   */
  async writeFile(filePath: string, content: string): Promise<FileOperationResult> {
    try {
      const validPath = validatePath(filePath);

      // Create parent directories if needed
      const parentDir = path.dirname(validPath);
      await fs.mkdir(parentDir, { recursive: true });

      // Write file
      await fs.writeFile(validPath, content, 'utf-8');

      // Get file size
      const stats = await fs.stat(validPath);

      return {
        success: true,
        path: validPath,
        size: stats.size,
      };
    } catch (error) {
      return {
        success: false,
        error: `Failed to write file: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  }

  /**
   * Delete a file or directory
   */
  async delete(targetPath: string): Promise<FileOperationResult> {
    try {
      const validPath = validatePath(targetPath);

      // Check if path exists
      try {
        await fs.access(validPath);
      } catch {
        throw new Error('Path not found');
      }

      // Get file info
      const stats = await fs.stat(validPath);

      // Delete
      if (stats.isFile()) {
        await fs.unlink(validPath);
      } else if (stats.isDirectory()) {
        await fs.rm(validPath, { recursive: true });
      }

      return {
        success: true,
        message: `Deleted ${path.basename(validPath)}`,
      };
    } catch (error) {
      return {
        success: false,
        error: `Failed to delete: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  }

  /**
   * Rename a file or directory
   */
  async rename(oldPath: string, newName: string): Promise<FileOperationResult> {
    try {
      const validOldPath = validatePath(oldPath);
      const newPath = path.join(path.dirname(validOldPath), newName);
      const validNewPath = validatePath(newPath);

      // Check if old path exists
      try {
        await fs.access(validOldPath);
      } catch {
        throw new Error('Path not found');
      }

      // Check if new path already exists
      try {
        await fs.access(validNewPath);
        throw new Error('Target name already exists');
      } catch (error) {
        // File doesn't exist, good to proceed (unless it's a different error)
        if (error instanceof Error && !error.message.includes('ENOENT')) {
          throw error;
        }
      }

      // Rename
      await fs.rename(validOldPath, validNewPath);

      return {
        success: true,
        message: `Renamed to ${newName}`,
        path: validNewPath,
      };
    } catch (error) {
      return {
        success: false,
        error: `Failed to rename: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  }

  /**
   * Create a directory
   */
  async createDirectory(parentPath: string, dirName: string): Promise<FileOperationResult> {
    try {
      const validParentPath = validatePath(parentPath);
      const newDirPath = path.join(validParentPath, dirName);
      const validNewPath = validatePath(newDirPath);

      // Check if directory already exists
      try {
        await fs.access(validNewPath);
        throw new Error('Directory already exists');
      } catch (error) {
        // Directory doesn't exist, good to proceed
        if (error instanceof Error && !error.message.includes('ENOENT')) {
          throw error;
        }
      }

      // Create directory
      await fs.mkdir(validNewPath, { recursive: true });

      return {
        success: true,
        path: validNewPath,
      };
    } catch (error) {
      return {
        success: false,
        error: `Failed to create directory: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  }

  /**
   * Check if a path exists
   */
  async exists(filePath: string): Promise<boolean> {
    try {
      const validPath = validatePath(filePath);
      await fs.access(validPath);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get file stats
   */
  async getStats(filePath: string): Promise<Stats | null> {
    try {
      const validPath = validatePath(filePath);
      return await fs.stat(validPath);
    } catch {
      return null;
    }
  }
}

// Export singleton instance
export const fileService = new FileService();
