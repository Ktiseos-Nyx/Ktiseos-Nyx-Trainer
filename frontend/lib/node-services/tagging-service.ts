/**
 * WD14 Tagging Service - Node.js ONNX Runtime Implementation
 *
 * Runs WD14 tagger entirely in Node.js using ONNX Runtime (CPU execution).
 * Eliminates Python subprocess spawning and asyncio issues.
 *
 * Based on Gemini's code with integration into existing job manager.
 */

import * as ort from 'onnxruntime-node';
import sharp from 'sharp';
import fs from 'fs/promises';
import path from 'path';
import https from 'https';
import { createWriteStream } from 'fs';

// ========== Types ==========

export interface TaggingConfig {
  dataset_dir: string;
  model: string;
  threshold: number;
  general_threshold?: number;
  character_threshold?: number;
  caption_extension: string;
  caption_separator: string;
  undesired_tags?: string;
  remove_underscore: boolean;
  character_tags_first: boolean;
  use_rating_tags: boolean;
  use_rating_tags_as_last_tag: boolean;
  append_tags: boolean;
}

export interface TagResult {
  tag: string;
  confidence: number;
}

// ========== Service Class ==========

class TaggingService {
  private session: ort.InferenceSession | null = null;
  private tags: string[] = [];
  private modelPath: string = '';
  private isInitialized: boolean = false;

  // Files needed for ONNX inference
  private static readonly REQUIRED_FILES = ['model.onnx', 'selected_tags.csv'];

  /**
   * Download a file from HuggingFace Hub
   */
  private async downloadFile(url: string, destPath: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const file = createWriteStream(destPath);
      const request = (reqUrl: string) => {
        https.get(reqUrl, (response) => {
          // Follow redirects (HuggingFace uses 301, 302, 303, 307, 308)
          if (response.statusCode && response.statusCode >= 300 && response.statusCode < 400) {
            const redirectUrl = response.headers.location;
            if (redirectUrl) {
              response.resume(); // Drain the response to free the socket
              request(redirectUrl);
              return;
            }
          }

          if (response.statusCode !== 200) {
            file.close();
            fs.unlink(destPath).catch(() => {});
            reject(new Error(`Download failed: HTTP ${response.statusCode} for ${url}`));
            return;
          }

          const totalSize = parseInt(response.headers['content-length'] || '0', 10);
          let downloaded = 0;

          response.on('data', (chunk: Buffer) => {
            downloaded += chunk.length;
            if (totalSize > 0 && downloaded % (5 * 1024 * 1024) < chunk.length) {
              const pct = ((downloaded / totalSize) * 100).toFixed(1);
              console.log(`[Tagging] Downloading... ${pct}% (${(downloaded / 1024 / 1024).toFixed(1)}MB)`);
            }
          });

          response.pipe(file);
          file.on('finish', () => {
            file.close();
            resolve();
          });
        }).on('error', (err) => {
          file.close();
          fs.unlink(destPath).catch(() => {});
          reject(err);
        });
      };
      request(url);
    });
  }

  /**
   * Download model files from HuggingFace if not already present
   */
  private async ensureModelDownloaded(modelName: string, modelDir: string): Promise<void> {
    await fs.mkdir(modelDir, { recursive: true });

    for (const file of TaggingService.REQUIRED_FILES) {
      const filePath = path.join(modelDir, file);
      try {
        await fs.access(filePath);
        const stat = await fs.stat(filePath);
        if (stat.size === 0) throw new Error('Empty file');
      } catch {
        const url = `https://huggingface.co/${modelName}/resolve/main/${file}`;
        console.log(`[Tagging] Downloading ${file} from HuggingFace: ${modelName}`);
        await this.downloadFile(url, filePath);
        console.log(`[Tagging] Downloaded ${file} successfully`);
      }
    }
  }

  /**
   * Initialize the WD14 model (run once per model)
   * Auto-downloads from HuggingFace if not present.
   */
  async initModel(modelName: string): Promise<void> {
    if (this.isInitialized && this.modelPath.includes(modelName)) {
      return; // Already initialized with this model
    }

    const projectRoot = path.resolve(process.cwd(), '..');
    const modelDir = path.join(projectRoot, 'tagger_models', modelName);
    const modelFile = path.join(modelDir, 'model.onnx');
    const csvFile = path.join(modelDir, 'selected_tags.csv');

    // Auto-download model files if missing
    await this.ensureModelDownloaded(modelName, modelDir);

    console.log(`[Tagging] Loading model: ${modelName}`);

    // Load ONNX Model (CPU execution - no CUDA needed!)
    this.session = await ort.InferenceSession.create(modelFile, {
      executionProviders: ['cpu'],
    });

    // Load Tags CSV
    const csvContent = await fs.readFile(csvFile, 'utf-8');
    this.tags = csvContent
      .split('\n')
      .slice(1) // Skip header
      .map((line) => {
        const parts = line.split(',');
        return parts[1]?.trim(); // Tag name is in column 1
      })
      .filter((t) => t); // Filter empty lines

    this.modelPath = modelFile;
    this.isInitialized = true;

    console.log(`[Tagging] Model loaded: ${this.tags.length} tags available`);
  }

  /**
   * Tag a single image
   */
  async tagImage(
    imagePath: string,
    threshold: number = 0.35,
    generalThreshold?: number,
    characterThreshold?: number
  ): Promise<TagResult[]> {
    if (!this.session) {
      throw new Error('Model not initialized. Call initModel() first.');
    }

    // --- PRE-PROCESSING ---
    // Resize to 448x448 with white padding, convert RGB to BGR
    const { data, info } = await sharp(imagePath)
      .removeAlpha() // Drop alpha channel
      .resize(448, 448, {
        fit: 'contain',
        background: { r: 255, g: 255, b: 255 }, // White padding
      })
      .raw()
      .toBuffer({ resolveWithObject: true });

    // Convert to Float32 and swap RGB -> BGR
    // WD14 expects BGR order with values 0-255 (not normalized)
    const float32Data = new Float32Array(448 * 448 * 3);

    for (let i = 0; i < 448 * 448; i++) {
      const r = data[i * 3 + 0];
      const g = data[i * 3 + 1];
      const b = data[i * 3 + 2];

      // Reverse to BGR
      float32Data[i * 3 + 0] = b;
      float32Data[i * 3 + 1] = g;
      float32Data[i * 3 + 2] = r;
    }

    // Create ONNX Tensor [1, 448, 448, 3] (NHWC format)
    const tensor = new ort.Tensor('float32', float32Data, [1, 448, 448, 3]);

    // --- INFERENCE ---
    const feeds = { [this.session.inputNames[0]]: tensor };
    const results = await this.session.run(feeds);

    // Get output scores
    const output = results[this.session.outputNames[0]].data as Float32Array;

    // --- POST-PROCESSING ---
    // Map scores to tags and filter by threshold
    const tagResults: TagResult[] = [];

    for (let i = 0; i < output.length; i++) {
      const score = output[i];
      const tag = this.tags[i];

      if (!tag) continue;

      // Apply category-specific thresholds
      let effectiveThreshold = threshold;

      // Check if it's a character tag (format: "character_name_(series)")
      const isCharacterTag = tag.includes('_(') && tag.includes(')');

      // Check if it's a general tag (most common category)
      // WD14 v3 has: rating, general, character tags
      // For simplicity, assume non-character tags are general
      const isGeneralTag = !isCharacterTag;

      if (isCharacterTag && characterThreshold !== undefined) {
        effectiveThreshold = characterThreshold;
      } else if (isGeneralTag && generalThreshold !== undefined) {
        effectiveThreshold = generalThreshold;
      }

      if (score >= effectiveThreshold) {
        tagResults.push({
          tag,
          confidence: score,
        });
      }
    }

    // Sort by confidence (highest first)
    tagResults.sort((a, b) => b.confidence - a.confidence);

    return tagResults;
  }

  /**
   * Tag all images in a dataset directory
   */
  async tagDataset(config: TaggingConfig, onProgress?: (current: number, total: number, filename: string) => void): Promise<{ success: boolean; processed: number; total: number; errors: string[] }> {
    // Initialize model
    await this.initModel(config.model);

    const datasetPath = path.resolve(process.cwd(), '..', config.dataset_dir);

    // Find all images
    const imageExtensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp'];
    const files = await fs.readdir(datasetPath);
    const imageFiles = files.filter((f) =>
      imageExtensions.some((ext) => f.toLowerCase().endsWith(ext))
    );

    console.log(`[Tagging] Found ${imageFiles.length} images in ${config.dataset_dir}`);

    let processedCount = 0;
    const errors: string[] = [];

    for (const imageFile of imageFiles) {
      const imagePath = path.join(datasetPath, imageFile);
      const captionPath = path.join(
        datasetPath,
        path.basename(imageFile, path.extname(imageFile)) + config.caption_extension
      );

      try {
        // Tag the image
        const tagResults = await this.tagImage(
          imagePath,
          config.threshold,
          config.general_threshold,
          config.character_threshold
        );

        // Post-process tags
        let tags = tagResults.map((r) => r.tag);

        // Remove underscores
        if (config.remove_underscore) {
          tags = tags.map((t) => t.replace(/_/g, ' '));
        }

        // Filter undesired tags
        if (config.undesired_tags) {
          const undesired = config.undesired_tags.split(',').map((t) => t.trim());
          tags = tags.filter((t) => !undesired.includes(t));
        }

        // Handle character tags first
        if (config.character_tags_first) {
          const characterTags = tags.filter((t) => t.includes('(') && t.includes(')'));
          const otherTags = tags.filter((t) => !characterTags.includes(t));
          tags = [...characterTags, ...otherTags];
        }

        // Append or overwrite
        if (config.append_tags) {
          try {
            const existingContent = await fs.readFile(captionPath, 'utf-8');
            const existingTags = existingContent.split(config.caption_separator).map((t) => t.trim());
            tags = [...new Set([...existingTags, ...tags])]; // Merge and dedupe
          } catch {
            // File doesn't exist, just write new tags
          }
        }

        // Write caption file
        const captionContent = tags.join(config.caption_separator);
        await fs.writeFile(captionPath, captionContent, 'utf-8');

        processedCount++;
        console.log(`[Tagging] ${processedCount}/${imageFiles.length} - ${imageFile}`);

        if (onProgress) {
          onProgress(processedCount, imageFiles.length, imageFile);
        }
      } catch (error) {
        const errorMsg = `Error processing ${imageFile}: ${error}`;
        console.error(`[Tagging] ${errorMsg}`);
        errors.push(errorMsg);
      }
    }

    console.log(`[Tagging] Completed: ${processedCount}/${imageFiles.length} images tagged`);

    return {
      success: errors.length === 0,
      processed: processedCount,
      total: imageFiles.length,
      errors,
    };
  }
}

// ========== Singleton Instance ==========

export const taggingService = new TaggingService();

// ========== Helper Function ==========

/**
 * Quick helper to tag a single image
 * (for testing or single-image operations)
 */
export async function tagSingleImage(
  imagePath: string,
  modelName: string = 'SmilingWolf/wd-vit-large-tagger-v3',
  threshold: number = 0.35
): Promise<TagResult[]> {
  await taggingService.initModel(modelName);
  return taggingService.tagImage(imagePath, threshold);
}
