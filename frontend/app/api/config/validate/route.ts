/**
 * POST /api/config/validate
 * Validate training configuration
 */

import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { config } = body;

    if (!config) {
      return NextResponse.json(
        { error: 'Missing configuration to validate' },
        { status: 400 }
      );
    }

    const errors: string[] = [];

    // Basic validation
    if (!config.pretrained_model_name_or_path) {
      errors.push('Missing required field: pretrained_model_name_or_path');
    }

    if (!config.train_data_dir) {
      errors.push('Missing required field: train_data_dir');
    }

    if (!config.output_dir) {
      errors.push('Missing required field: output_dir');
    }

    if (!config.output_name) {
      errors.push('Missing required field: output_name');
    }

    if (!config.resolution || config.resolution < 256) {
      errors.push('Invalid resolution: must be at least 256');
    }

    if (!config.train_batch_size || config.train_batch_size < 1) {
      errors.push('Invalid batch size: must be at least 1');
    }

    if (!config.max_train_epochs && !config.max_train_steps) {
      errors.push('Must specify either max_train_epochs or max_train_steps');
    }

    return NextResponse.json({
      valid: errors.length === 0,
      errors,
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
