/**
 * GET /api/config/defaults
 * Get default training configuration
 */

import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const defaultConfig = {
    // Model
    model_type: 'SDXL',
    pretrained_model_name_or_path: '',
    vae_path: '',
    
    // Paths
    train_data_dir: './dataset',
    output_dir: './output',
    output_name: 'my_lora',
    logging_dir: './logs',
    
    // Training
    max_train_epochs: 10,
    train_batch_size: 1,
    resolution: 1024,
    seed: 42,
    
    // Learning Rate
    unet_lr: 1e-4,
    text_encoder_lr: 1e-5,
    lr_scheduler: 'constant',
    lr_warmup_steps: 0,
    
    // Optimizer
    optimizer_type: 'AdamW8bit',
    
    // LoRA
    network_module: 'networks.lora',
    network_dim: 32,
    network_alpha: 16,
    
    // Saving
    save_every_n_epochs: 1,
    save_model_as: 'safetensors',
    
    // Mixed Precision
    mixed_precision: 'bf16',
    
    // Cache
    cache_latents: true,
    cache_latents_to_disk: true,
  };

  return NextResponse.json({
    config: defaultConfig,
  });
}
