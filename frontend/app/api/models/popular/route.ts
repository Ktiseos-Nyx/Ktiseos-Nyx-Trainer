/**
 * Next.js API Route: GET /api/models/popular
 * Get a list of popular/recommended models and VAEs with direct download URLs
 *
 * Migrated from Python FastAPI: api/routes/models.py -> get_popular_models
 */

import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    success: true,
    models: {
      sdxl: [
        {
          name: 'SDXL Base 1.0',
          url: 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors',
          filename: 'sd_xl_base_1.0.safetensors',
          description: 'Official SDXL base model from Stability AI',
        },
        {
          name: 'Illustrious XL',
          url: 'https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors',
          filename: 'Illustrious-XL-v0.1.safetensors',
          description: 'High-quality anime/illustration model',
        },
        {
          name: 'Pony Diffusion V6 XL',
          url: 'https://civitai.com/api/download/models/290640',
          filename: 'ponyDiffusionV6XL_v6.safetensors',
          description: 'Popular anime/cartoon model',
        },
      ],
      sd15: [
        {
          name: 'SD 1.5',
          url: 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors',
          filename: 'v1-5-pruned-emaonly.safetensors',
          description: 'Official SD 1.5 model',
        },
      ],
      flux: [
        {
          name: 'FLUX.1 Dev',
          url: 'https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors',
          filename: 'flux1-dev.safetensors',
          description: 'FLUX.1 development model',
        },
      ],
    },
    vaes: [
      {
        name: 'SDXL VAE',
        url: 'https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors',
        filename: 'sdxl_vae.safetensors',
        description: 'Official SDXL VAE',
      },
      {
        name: 'SD 1.5 VAE (MSE)',
        url: 'https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors',
        filename: 'vae-ft-mse-840000-ema-pruned.safetensors',
        description: 'Improved VAE for SD 1.5',
      },
    ],
  });
}
