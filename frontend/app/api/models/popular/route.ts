/**
 * Next.js API Route: GET /api/models/popular
 * Get a list of supported models and VAEs with direct download URLs
 *
 * Migrated from Python FastAPI: api/routes/models.py -> get_popular_models
 */

import { NextResponse } from 'next/server';

/**
 * Provide a fixed JSON payload listing supported models and VAEs with download metadata.
 *
 * The response contains:
 * - `success`: boolean set to `true`.
 * - `models`: an object mapping category keys (e.g., `sdxl`, `sd15`, `flux`, `sd3.5`, `chroma`, `anima`, `lumina`, `hunyuanimage`) to arrays of model entries. Each model entry includes `name`, `url`, `filename`, and `description`; some entries may include `manualOnly: true` and `repoUrl` for manual-download guidance or gated access notes.
 * - `vaes`: an array of VAE entries, each including `name`, `url`, `filename`, and `description`.
 *
 * @returns A JSON object with `success`, `models`, and `vaes` describing available models and VAEs and their download or manual-download instructions.
 */
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
          description: 'Anime/cartoon model from Civitai',
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
          name: 'FLUX.1 Dev (Gated)',
          url: 'https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors',
          filename: 'flux1-dev.safetensors',
          description: 'FLUX.1 development model — requires HuggingFace login & license acceptance',
        },
        {
          name: 'FLUX.1 Schnell',
          url: 'https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors',
          filename: 'flux1-schnell.safetensors',
          description: 'FLUX.1 fast inference model — Apache 2.0 license, no login required',
        },
      ],
      'sd3.5': [
        {
          name: 'SD 3.5 Large (Gated)',
          url: 'https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/sd3.5_large.safetensors',
          filename: 'sd3.5_large.safetensors',
          description: 'Stability AI SD 3.5 Large — may require HuggingFace login & license acceptance',
        },
      ],
      chroma: [
        {
          name: 'Chroma1 Base',
          url: 'https://huggingface.co/lodestones/Chroma1-Base/resolve/main/Chroma1-Base.safetensors',
          filename: 'Chroma1-Base.safetensors',
          description: 'Chroma base model by Lodestone — no CLIP-L needed, T5-XXL only',
        },
        {
          name: 'Chroma1 HD',
          url: 'https://huggingface.co/lodestones/Chroma1-HD/resolve/main/Chroma1-HD.safetensors',
          filename: 'Chroma1-HD.safetensors',
          description: 'Chroma HD model by Lodestone — higher resolution variant',
        },
      ],
      anima: [
        {
          name: 'Anima Preview (Diffusion Model)',
          url: 'https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-preview.safetensors',
          filename: 'anima-preview.safetensors',
          description: 'Anima diffusion model by Circlestone Labs — Qwen3 + T5 dual encoder architecture',
        },
        {
          name: 'Anima Text Encoder (Qwen3 0.6B)',
          url: 'https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors',
          filename: 'qwen_3_06b_base.safetensors',
          description: 'Qwen3 0.6B text encoder for Anima — required component',
        },
      ],
      lumina: [
        {
          name: 'Lumina Image 2.0 (Diffusion Model)',
          url: 'https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/diffusion_models/lumina_2_model_bf16.safetensors',
          filename: 'lumina_2_model_bf16.safetensors',
          description: 'Lumina Image 2.0 diffusion model (5.2GB) — 2B parameter flow-based DiT',
        },
        {
          name: 'Lumina Gemma2 Text Encoder',
          url: 'https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/text_encoders/gemma_2_2b_fp16.safetensors',
          filename: 'gemma_2_2b_fp16.safetensors',
          description: 'Gemma2 2B text encoder for Lumina — required component (5.2GB)',
        },
      ],
      hunyuanimage: [
        {
          name: 'HunyuanImage 3.0 (Manual Download)',
          url: '',
          filename: '',
          description: '168GB model (32 sharded files) — too large for web download. Run: huggingface-cli download tencent/HunyuanImage-3.0 --local-dir ./pretrained_model/hunyuanimage',
          manualOnly: true,
          repoUrl: 'https://huggingface.co/tencent/HunyuanImage-3.0',
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
      {
        name: 'Flux VAE',
        url: 'https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors',
        filename: 'flux_ae.safetensors',
        description: 'Flux autoencoder (335MB) — same VAE for both Dev and Schnell, from ungated Schnell repo',
      },
      {
        name: 'Anima VAE',
        url: 'https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors',
        filename: 'qwen_image_vae.safetensors',
        description: 'Anima-specific VAE — does NOT use Flux VAE',
      },
      {
        name: 'Chroma VAE',
        url: 'https://huggingface.co/lodestones/Chroma1-Base/resolve/main/vae/diffusion_pytorch_model.safetensors',
        filename: 'chroma_vae.safetensors',
        description: 'Chroma VAE from Lodestone — included in Chroma repos',
      },
      {
        name: 'Lumina VAE',
        url: 'https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors',
        filename: 'lumina_ae.safetensors',
        description: 'Lumina Image 2.0 autoencoder (335MB)',
      },
    ],
  });
}
