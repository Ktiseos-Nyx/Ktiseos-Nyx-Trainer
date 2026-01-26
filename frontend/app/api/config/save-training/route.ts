/**
 * POST /api/config/save-training
 * Generate and save training configuration (dataset.toml + config.toml)
 * This is the CRITICAL route that uses config-service.ts
 */

import { NextRequest, NextResponse } from 'next/server';
import { ConfigService } from '@/lib/node-services/config-service';

export async function POST(request: NextRequest) {
  try {
    const config = await request.json();

    if (!config) {
      return NextResponse.json(
        { error: 'Missing training configuration' },
        { status: 400 }
      );
    }

    // Initialize config service
    const configService = new ConfigService();

    // Generate TOMLs using the service
    const result = await configService.generateTOMLs(config);

    if (!result.success) {
      return NextResponse.json(
        {
          success: false,
          errors: result.errors,
          message: 'Failed to generate training configuration',
        },
        { status: 400 }
      );
    }

    return NextResponse.json({
      success: true,
      dataset_config_path: result.datasetPath,
      training_config_path: result.configPath,
      message: 'Training configuration generated successfully',
    });
  } catch (error) {
    console.error('Failed to save training config:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
