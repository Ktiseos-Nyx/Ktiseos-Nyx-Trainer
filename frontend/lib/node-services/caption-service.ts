/**
 * Caption Service - Node.js caption editing operations
 * Migrated from Python services/caption_service.py
 *
 * Handles:
 * - Add trigger words to captions
 * - Remove specific tags
 * - Replace text (with regex support)
 * - Read/write individual captions
 * - Bulk operations on entire datasets
 */

import fs from 'fs/promises';
import path from 'path';

// ========== Types ==========

export interface AddTriggerWordRequest {
  dataset_dir: string;
  trigger_word: string;
  position: 'start' | 'end';
  caption_extension?: string;
}

export interface RemoveTagsRequest {
  dataset_dir: string;
  tags_to_remove: string[];
  caption_extension?: string;
}

export interface ReplaceTextRequest {
  dataset_dir: string;
  find_text: string;
  replace_text: string;
  use_regex?: boolean;
  caption_extension?: string;
}

export interface ReadCaptionRequest {
  image_path: string;
  caption_extension?: string;
}

export interface WriteCaptionRequest {
  image_path: string;
  caption_text: string;
  caption_extension?: string;
}

export interface CaptionOperationResponse {
  success: boolean;
  message: string;
  files_modified: number;
  errors: string[];
}

export interface CaptionReadResponse {
  success: boolean;
  image_path: string;
  caption_path: string;
  caption_text: string | null;
  exists: boolean;
}

// ========== Configuration ==========

const ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'];
const DEFAULT_CAPTION_EXTENSION = '.txt';

// ========== Helper Functions ==========

/**
 * Validate dataset path exists and is a directory
 */
async function validateDatasetPath(datasetPath: string): Promise<string> {
  try {
    const stats = await fs.stat(datasetPath);
    if (!stats.isDirectory()) {
      throw new Error('Path is not a directory');
    }
    return path.resolve(datasetPath);
  } catch (error) {
    if (error instanceof Error && error.message.includes('ENOENT')) {
      throw new Error(`Dataset not found: ${datasetPath}`);
    }
    throw error;
  }
}

/**
 * Check if file is an image based on extension
 */
function isImageFile(filePath: string): boolean {
  const ext = path.extname(filePath).toLowerCase();
  return ALLOWED_IMAGE_EXTENSIONS.includes(ext);
}

/**
 * Get all image files in a directory
 */
async function getImageFiles(dirPath: string): Promise<string[]> {
  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  return entries
    .filter(entry => entry.isFile() && isImageFile(entry.name))
    .map(entry => path.join(dirPath, entry.name));
}

/**
 * Get all caption files in a directory
 */
async function getCaptionFiles(dirPath: string, extension: string): Promise<string[]> {
  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  return entries
    .filter(entry => entry.isFile() && entry.name.endsWith(extension))
    .map(entry => path.join(dirPath, entry.name));
}

// ========== CaptionService Class ==========

export class CaptionService {
  /**
   * Add a trigger word to all captions in a dataset
   */
  async addTriggerWord(request: AddTriggerWordRequest): Promise<CaptionOperationResponse> {
    try {
      const datasetPath = await validateDatasetPath(request.dataset_dir);
      const captionExt = request.caption_extension || DEFAULT_CAPTION_EXTENSION;

      let filesModified = 0;
      const errors: string[] = [];

      // Get all image files
      const imageFiles = await getImageFiles(datasetPath);

      for (const imageFile of imageFiles) {
        try {
          // Get corresponding caption file
          const captionFile = imageFile.replace(path.extname(imageFile), captionExt);

          // Read existing caption or create empty
          let captionText = '';
          try {
            captionText = (await fs.readFile(captionFile, 'utf-8')).trim();
          } catch {
            // Caption file doesn't exist, start with empty string
          }

          // Add trigger word
          let newCaption: string;
          if (request.position === 'start') {
            newCaption = captionText
              ? `${request.trigger_word}, ${captionText}`
              : request.trigger_word;
          } else {
            // end
            newCaption = captionText
              ? `${captionText}, ${request.trigger_word}`
              : request.trigger_word;
          }

          // Write back
          await fs.writeFile(captionFile, newCaption, 'utf-8');
          filesModified++;
        } catch (error) {
          errors.push(
            `${path.basename(imageFile)}: ${error instanceof Error ? error.message : String(error)}`
          );
        }
      }

      return {
        success: true,
        message: `Added trigger word to ${filesModified} captions`,
        files_modified: filesModified,
        errors,
      };
    } catch (error) {
      return {
        success: false,
        message: error instanceof Error ? error.message : String(error),
        files_modified: 0,
        errors: [error instanceof Error ? error.message : String(error)],
      };
    }
  }

  /**
   * Remove specific tags from all captions in a dataset
   */
  async removeTags(request: RemoveTagsRequest): Promise<CaptionOperationResponse> {
    try {
      const datasetPath = await validateDatasetPath(request.dataset_dir);
      const captionExt = request.caption_extension || DEFAULT_CAPTION_EXTENSION;

      let filesModified = 0;
      const errors: string[] = [];

      // Get all caption files
      const captionFiles = await getCaptionFiles(datasetPath, captionExt);

      // Create lowercase set for case-insensitive matching
      const tagsToRemoveLower = new Set(request.tags_to_remove.map(tag => tag.toLowerCase()));

      for (const captionFile of captionFiles) {
        try {
          const captionText = (await fs.readFile(captionFile, 'utf-8')).trim();

          // Split into tags
          const tags = captionText.split(',').map(tag => tag.trim());

          // Remove unwanted tags (case-insensitive)
          const filteredTags = tags.filter(tag => !tagsToRemoveLower.has(tag.toLowerCase()));

          // Write back if changed
          if (filteredTags.length !== tags.length) {
            const newCaption = filteredTags.join(', ');
            await fs.writeFile(captionFile, newCaption, 'utf-8');
            filesModified++;
          }
        } catch (error) {
          errors.push(
            `${path.basename(captionFile)}: ${error instanceof Error ? error.message : String(error)}`
          );
        }
      }

      return {
        success: true,
        message: `Removed tags from ${filesModified} captions`,
        files_modified: filesModified,
        errors,
      };
    } catch (error) {
      return {
        success: false,
        message: error instanceof Error ? error.message : String(error),
        files_modified: 0,
        errors: [error instanceof Error ? error.message : String(error)],
      };
    }
  }

  /**
   * Replace text in all captions (with optional regex support)
   */
  async replaceText(request: ReplaceTextRequest): Promise<CaptionOperationResponse> {
    try {
      const datasetPath = await validateDatasetPath(request.dataset_dir);
      const captionExt = request.caption_extension || DEFAULT_CAPTION_EXTENSION;

      let filesModified = 0;
      const errors: string[] = [];

      // Get all caption files
      const captionFiles = await getCaptionFiles(datasetPath, captionExt);

      for (const captionFile of captionFiles) {
        try {
          const captionText = await fs.readFile(captionFile, 'utf-8');

          // Replace text
          let newCaption: string;
          if (request.use_regex) {
            const regex = new RegExp(request.find_text, 'g');
            newCaption = captionText.replace(regex, request.replace_text);
          } else {
            newCaption = captionText.split(request.find_text).join(request.replace_text);
          }

          // Write back if changed
          if (newCaption !== captionText) {
            await fs.writeFile(captionFile, newCaption, 'utf-8');
            filesModified++;
          }
        } catch (error) {
          errors.push(
            `${path.basename(captionFile)}: ${error instanceof Error ? error.message : String(error)}`
          );
        }
      }

      return {
        success: true,
        message: `Replaced text in ${filesModified} captions`,
        files_modified: filesModified,
        errors,
      };
    } catch (error) {
      return {
        success: false,
        message: error instanceof Error ? error.message : String(error),
        files_modified: 0,
        errors: [error instanceof Error ? error.message : String(error)],
      };
    }
  }

  /**
   * Read a single caption file
   */
  async readCaption(request: ReadCaptionRequest): Promise<CaptionReadResponse> {
    try {
      const imagePath = path.resolve(request.image_path);
      const captionExt = request.caption_extension || DEFAULT_CAPTION_EXTENSION;
      const captionPath = imagePath.replace(path.extname(imagePath), captionExt);

      try {
        const captionText = await fs.readFile(captionPath, 'utf-8');
        return {
          success: true,
          image_path: imagePath,
          caption_path: captionPath,
          caption_text: captionText,
          exists: true,
        };
      } catch {
        // Caption file doesn't exist
        return {
          success: true,
          image_path: imagePath,
          caption_path: captionPath,
          caption_text: null,
          exists: false,
        };
      }
    } catch (error) {
      return {
        success: false,
        image_path: request.image_path,
        caption_path: path.resolve(request.image_path).replace(path.extname(request.image_path), request.caption_extension || DEFAULT_CAPTION_EXTENSION),
        caption_text: null,
        exists: false,
      };
    }
  }

  /**
   * Write a single caption file
   */
  async writeCaption(request: WriteCaptionRequest): Promise<CaptionOperationResponse> {
    try {
      const imagePath = path.resolve(request.image_path);
      const captionExt = request.caption_extension || DEFAULT_CAPTION_EXTENSION;
      const captionPath = imagePath.replace(path.extname(imagePath), captionExt);

      // Ensure parent directory exists
      const parentDir = path.dirname(captionPath);
      await fs.mkdir(parentDir, { recursive: true });

      // Write caption
      await fs.writeFile(captionPath, request.caption_text, 'utf-8');

      return {
        success: true,
        message: `Caption written to ${path.basename(captionPath)}`,
        files_modified: 1,
        errors: [],
      };
    } catch (error) {
      return {
        success: false,
        message: `Failed to write caption: ${error instanceof Error ? error.message : String(error)}`,
        files_modified: 0,
        errors: [error instanceof Error ? error.message : String(error)],
      };
    }
  }
}

// Export singleton instance
export const captionService = new CaptionService();
