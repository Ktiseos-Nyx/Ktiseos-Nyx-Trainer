/**
 * Next.js API Route: POST /api/utilities/calculator
 * Calculate training steps using Kohya-compatible logic
 *
 * Migrated from Python FastAPI: api/routes/utilities.py -> calculate_steps
 */

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

const ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'];

interface CalculatorRequest {
  dataset_path: string;
  epochs?: number;
  batch_size?: number;
}

interface CalculatorResponse {
  success: boolean;
  dataset_path: string;
  images: number;
  repeats: number;
  epochs: number;
  batch_size: number;
  total_steps: number;
  caption: string | null;
  time_estimate_min: number;
  time_estimate_max: number;
  recommendation: string;
}

/**
 * Extract Kohya parameters from folder name
 * Format: "N_caption" where N is repeat count
 */
function extractKohyaParams(folderName: string): { repeats: number; caption: string | null } {
  // Match pattern like "5_woman" or "10_my_character"
  const match = folderName.match(/^(\d+)_(.+)$/);

  if (match) {
    return {
      repeats: parseInt(match[1], 10),
      caption: match[2].replace(/_/g, ' '),
    };
  }

  // Default if no prefix
  return {
    repeats: 1,
    caption: null,
  };
}

/**
 * Count images in a directory
 */
async function countImages(dirPath: string): Promise<number> {
  try {
    const entries = await fs.readdir(dirPath, { withFileTypes: true });
    return entries.filter(entry => {
      if (!entry.isFile()) return false;
      const ext = path.extname(entry.name).toLowerCase();
      return ALLOWED_IMAGE_EXTENSIONS.includes(ext);
    }).length;
  } catch {
    return 0;
  }
}

export async function POST(request: NextRequest) {
  try {
    const body: CalculatorRequest = await request.json();
    const { dataset_path, epochs = 10, batch_size = 1 } = body;

    if (!dataset_path || !dataset_path.trim()) {
      return NextResponse.json(
        { error: 'Dataset path is required' },
        { status: 400 }
      );
    }

    if (batch_size <= 0) {
      return NextResponse.json(
        { error: 'Batch size must be greater than zero' },
        { status: 400 }
      );
    }

    // Check if path exists
    try {
      const stats = await fs.stat(dataset_path);
      if (!stats.isDirectory()) {
        return NextResponse.json(
          { error: 'Path is not a directory' },
          { status: 400 }
        );
      }
    } catch {
      return NextResponse.json(
        { error: `Dataset path does not exist: ${dataset_path}` },
        { status: 404 }
      );
    }

    // Extract Kohya parameters from folder name
    const folderName = path.basename(dataset_path);
    const { repeats, caption } = extractKohyaParams(folderName);

    // Count images
    const images = await countImages(dataset_path);

    if (images === 0) {
      return NextResponse.json(
        { error: `No images found in: ${dataset_path}` },
        { status: 400 }
      );
    }

    // Calculate total steps using Kohya's exact logic
    const totalSteps = Math.floor((images * repeats * epochs) / batch_size);

    // Time estimation (approximate)
    const timeEstimateMin = (totalSteps * 2) / 60; // GPU rental (faster)
    const timeEstimateMax = (totalSteps * 4) / 60; // Home GPU

    // Recommendation
    let recommendation: string;
    if (totalSteps < 500) {
      recommendation = '⚠️ Low step count - may underfit. Consider more epochs or repeats.';
    } else if (totalSteps > 5000) {
      recommendation = '⚠️ High step count - may overfit. Consider fewer epochs.';
    } else {
      recommendation = '✅ Good step count for most LoRA training scenarios.';
    }

    const response: CalculatorResponse = {
      success: true,
      dataset_path,
      images,
      repeats,
      epochs,
      batch_size,
      total_steps: totalSteps,
      caption,
      time_estimate_min: Math.round(timeEstimateMin * 100) / 100,
      time_estimate_max: Math.round(timeEstimateMax * 100) / 100,
      recommendation,
    };

    return NextResponse.json(response);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: errorMessage, detail: errorMessage },
      { status: 500 }
    );
  }
}
