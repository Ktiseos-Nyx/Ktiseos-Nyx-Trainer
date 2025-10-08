# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

import logging
import os
import subprocess
from pathlib import Path

from .managers import get_subprocess_environment, get_venv_python_path

logger = logging.getLogger(__name__)

def detect_model_type(model_path: str) -> str:
    """
    Detect if a model is SD 1.5 or SDXL based on file size and naming conventions.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        str: 'sdxl' or 'sd15'
    """
    model_path = Path(model_path)

    # First check filename patterns (most reliable)
    model_name = model_path.name.lower()

    # SDXL indicators in filename
    sdxl_indicators = ['sdxl', 'xl', 'illustrious', 'pony', 'noobai']
    if any(indicator in model_name for indicator in sdxl_indicators):
        return 'sdxl'

    # SD 1.5 indicators in filename
    sd15_indicators = ['sd15', 'v1-5', 'anything', 'counterfeit', 'realistic']
    if any(indicator in model_name for indicator in sd15_indicators):
        return 'sd15'

    # Fall back to file size estimation (rough heuristic)
    try:
        file_size_gb = model_path.stat().st_size / (1024**3)

        # SDXL models are typically 6-7GB, SD 1.5 models are typically 2-4GB
        if file_size_gb > 5.5:
            logger.info(f"  📏 Model size {file_size_gb:.1f}GB suggests SDXL")
            return 'sdxl'
        else:
            logger.info(f"  📏 Model size {file_size_gb:.1f}GB suggests SD 1.5")
            return 'sd15'
    except Exception as e:
        # If we can't determine file size, default to SD 1.5 (more common)
        logger.warning(f"  ⚠️  Could not determine model type ({e}), defaulting to SD 1.5")
        return 'sd15'

def generate_sd15_samples(lora_path: str, base_model_path: str, prompt: str, num_samples: int,
                         output_dir: str, epoch: int, **kwargs):
    """Generate samples for SD 1.5 models using gen_img_diffusers.py or gen_img.py"""
    project_root = Path(__file__).parents[1]
    sd_scripts_dir = project_root / "trainer" / "derrian_backend" / "sd_scripts"

    # Try diffusers version first, fallback to original
    inference_script = sd_scripts_dir / "gen_img_diffusers.py"
    if not inference_script.exists():
        inference_script = sd_scripts_dir / "gen_img.py"

    if not inference_script.exists():
        logger.error("❌ SD 1.5 inference script not found!")
        return False

    logger.info(f"🎨 Using SD 1.5 inference: {inference_script.name}")
    # SD 1.5 specific inference logic here
    return _run_inference_script(inference_script, lora_path, base_model_path, prompt,
                                num_samples, output_dir, epoch, resolution=512, **kwargs)

def generate_sdxl_samples(lora_path: str, base_model_path: str, prompt: str, num_samples: int,
                         output_dir: str, epoch: int, **kwargs):
    """Generate samples for SDXL models using sdxl_gen_img.py or sdxl_minimal_inference.py"""
    project_root = Path(__file__).parents[1]
    sd_scripts_dir = project_root / "trainer" / "derrian_backend" / "sd_scripts"

    # Try sdxl_gen_img.py first, fallback to sdxl_minimal_inference.py
    inference_script = sd_scripts_dir / "sdxl_gen_img.py"
    if not inference_script.exists():
        inference_script = sd_scripts_dir / "sdxl_minimal_inference.py"

    if not inference_script.exists():
        logger.error("❌ SDXL inference script not found! Looked for:")
        logger.error("   - sdxl_gen_img.py")
        logger.error("   - sdxl_minimal_inference.py")
        return False

    logger.info(f"🎨 Using SDXL inference: {inference_script.name}")
    # SDXL specific inference logic here
    return _run_inference_script(inference_script, lora_path, base_model_path, prompt,
                                num_samples, output_dir, epoch, resolution=1024, **kwargs)

def generate_flux_samples(lora_path: str, base_model_path: str, prompt: str, num_samples: int,
                         output_dir: str, epoch: int, **kwargs):
    """Generate samples for Flux models using flux_minimal_inference.py"""
    project_root = Path(__file__).parents[1]
    sd_scripts_dir = project_root / "trainer" / "derrian_backend" / "sd_scripts"

    inference_script = sd_scripts_dir / "flux_minimal_inference.py"

    if not inference_script.exists():
        logger.error(f"❌ Flux inference script not found at: {inference_script}")
        return False

    logger.info(f"🎨 Using Flux inference: {inference_script.name}")
    # Flux specific inference logic here
    return _run_inference_script(inference_script, lora_path, base_model_path, prompt,
                                num_samples, output_dir, epoch, resolution=1024, **kwargs)

def generate_sd3_samples(lora_path: str, base_model_path: str, prompt: str, num_samples: int,
                        output_dir: str, epoch: int, **kwargs):
    """Generate samples for SD3 models using sd3_minimal_inference.py"""
    project_root = Path(__file__).parents[1]
    sd_scripts_dir = project_root / "trainer" / "derrian_backend" / "sd_scripts"

    inference_script = sd_scripts_dir / "sd3_minimal_inference.py"

    if not inference_script.exists():
        logger.error(f"❌ SD3 inference script not found at: {inference_script}")
        return False

    logger.info(f"🎨 Using SD3 inference: {inference_script.name}")
    # SD3 specific inference logic here
    return _run_inference_script(inference_script, lora_path, base_model_path, prompt,
                                num_samples, output_dir, epoch, resolution=1024, **kwargs)

def generate_sample_images(
    lora_path: str,
    base_model_path: str,
    prompt: str,
    num_samples: int,
    output_dir: str,
    epoch: int,
    resolution: int = None,
    seed: int = 42,
    negative_prompt: str = "bad anatomy, bad hands, missing fingers, extra fingers, fewer fingers, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    guidance_scale: float = 7.5,
    num_inference_steps: int = 25,
):
    """
    Auto-detects model type and routes to appropriate inference function.
    """
    logger.info(f"✨ Generating {num_samples} sample images for epoch {epoch}...")

    # Detect model type and route to appropriate function
    model_type = detect_model_type(base_model_path)
    logger.info(f"🔍 Detected model type: {model_type.upper()}")

    kwargs = {
        'resolution': resolution,
        'seed': seed,
        'negative_prompt': negative_prompt,
        'guidance_scale': guidance_scale,
        'num_inference_steps': num_inference_steps
    }

    if model_type == 'sd15':
        return generate_sd15_samples(lora_path, base_model_path, prompt, num_samples, output_dir, epoch, **kwargs)
    elif model_type == 'sdxl':
        return generate_sdxl_samples(lora_path, base_model_path, prompt, num_samples, output_dir, epoch, **kwargs)
    elif model_type == 'flux':
        return generate_flux_samples(lora_path, base_model_path, prompt, num_samples, output_dir, epoch, **kwargs)
    elif model_type == 'sd3':
        return generate_sd3_samples(lora_path, base_model_path, prompt, num_samples, output_dir, epoch, **kwargs)
    else:
        logger.error(f"❌ Unsupported model type: {model_type}")
        return False

def _run_inference_script(inference_script, lora_path, base_model_path, prompt, num_samples,
                         output_dir, epoch, **kwargs):
    """Common inference script execution logic"""
    
    # Extract parameters from kwargs
    resolution = kwargs.get('resolution', 512)
    seed = kwargs.get('seed', 42) 
    negative_prompt = kwargs.get('negative_prompt', 'bad anatomy, bad hands')
    guidance_scale = kwargs.get('guidance_scale', 7.5)
    num_inference_steps = kwargs.get('num_inference_steps', 25)
    
    # Get project root and SD scripts directory
    project_root = Path(__file__).parents[1]
    sd_scripts_dir = project_root / "trainer" / "derrian_backend" / "sd_scripts"

    # Get proper venv python and environment (fixes CAME import issues!)
    venv_python = get_venv_python_path(str(project_root))
    if not os.path.exists(venv_python):
        logger.warning(f"⚠️ Virtual environment python not found at {venv_python}, using system python")
        venv_python = "python"

    # Get standardized subprocess environment
    env = get_subprocess_environment(str(project_root))

    # Create a specific output directory for this epoch's samples
    epoch_sample_output_dir = Path(output_dir) / "sample_images" / f"epoch_{epoch:03d}"
    epoch_sample_output_dir.mkdir(parents=True, exist_ok=True)

    # Base command arguments (now using proper venv python!)
    cmd_args = [
        venv_python,
        str(inference_script),
        "--ckpt_path", str(base_model_path),
        "--prompt", prompt,
        "--negative_prompt", negative_prompt,
        "--output_dir", str(epoch_sample_output_dir),
        "--steps", str(num_inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--target_height", str(resolution),
        "--target_width", str(resolution),
        "--original_height", str(resolution),
        "--original_width", str(resolution),
    ]

    # Add LoRA weights if provided
    if lora_path and Path(lora_path).exists():
        # sdxl_minimal_inference.py expects 'path;multiplier'
        # We'll use a default multiplier of 1.0
        cmd_args.extend(["--lora_weights", f"{lora_path};1.0"])
    else:
        logger.warning(f"⚠️ LoRA checkpoint not found at {lora_path}. Generating samples without LoRA.")

    # Loop to generate multiple samples
    for i in range(num_samples):
        current_seed = seed + i # Vary seed for different samples
        sample_cmd = cmd_args + ["--seed", str(current_seed)]
        logger.info(f"  Running inference command (sample {i+1}/{num_samples}): {' '.join(map(str, sample_cmd))}")

        try:
            # Run the subprocess with proper environment and working directory
            process = subprocess.run(
                sample_cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
                cwd=str(sd_scripts_dir)  # Run from scripts directory for proper imports
            )
            logger.info(f"  Inference output (sample {i+1}):\n{process.stdout}")
            if process.stderr:
                logger.warning(f"  Inference stderr (sample {i+1}):\n{process.stderr}")

        except subprocess.CalledProcessError as e:
            logger.error(f"🚨 Error generating sample image {i+1} for epoch {epoch}: {e}")
            logger.error(f"  Stdout: {e.stdout}")
            logger.error(f"  Stderr: {e.stderr}")
            break # Stop if one sample fails
        except Exception as e:
            logger.error(f"🚨 Unexpected error during sample generation {i+1} for epoch {epoch}: {e}")
            break # Stop if one sample fails

    logger.info(f"✅ Finished generating samples for epoch {epoch}.")
