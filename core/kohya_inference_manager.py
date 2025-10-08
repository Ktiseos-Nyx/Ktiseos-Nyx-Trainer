
# core/kohya_inference_manager.py
"""
Unified Inference Manager leveraging Kohya's library for image generation.
"""
import os
import sys
import subprocess
from typing import Any, Dict

# Add Kohya's backend to system path
sys.path.insert(0, os.path.join(os.getcwd(), "trainer", "derrian_backend", "sd_scripts"))

import logging

from library import model_util
from library.strategy_sd import SdTextEncodingStrategy, SdTokenizeStrategy
from library.strategy_sdxl import (SdxlTextEncodingStrategy,
                                   SdxlTokenizeStrategy)

# Import other strategies as needed, e.g., flux, sd3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KohyaInferenceManager:
    def __init__(self):
        self.strategies = {
            'sd15': {'tokenize': SdTokenizeStrategy(v2=False, max_length=75), 'encoding': SdTextEncodingStrategy(clip_skip=1)},
            'sdxl': {'tokenize': SdxlTokenizeStrategy(max_length=75), 'encoding': SdxlTextEncodingStrategy()},
            # 'flux': {'tokenize': FluxTokenizeStrategy, 'encoding': FluxTextEncodingStrategy},
            # 'sd3': {'tokenize': Sd3TokenizeStrategy, 'encoding': Sd3TextEncodingStrategy}
        }
        self.models_cache = {}

    def detect_model_type(self, model_path: str) -> str:
        """
        Detects the model type (e.g., 'sd15', 'sdxl') from the model path.
        """
        model_path_lower = model_path.lower()
        if 'sdxl' in model_path_lower or 'xl' in model_path_lower:
            return 'sdxl'
        # Add more sophisticated detection logic as needed
        return 'sd15'

    def _load_model(self, model_path: str, model_type: str):
        """
        Loads the model, text_encoder, and VAE from a checkpoint.
        Caches the loaded models to avoid reloading.
        """
        if model_path in self.models_cache:
            return self.models_cache[model_path]

        logger.info(f"Loading {model_type} model from: {model_path}")

        v2 = 'v2' in model_path.lower()

        text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(
            v2=v2,
            ckpt_path=model_path
        )

        self.models_cache[model_path] = (text_encoder, vae, unet)
        return text_encoder, vae, unet

    def generate_image(self, params: Dict[str, Any]):
        """
        Generates an image based on the provided parameters.
        """
        model_path = params.get('model_path')
        prompt = params.get('prompt', '')

        if not model_path or not os.path.exists(model_path):
            logger.error("Model path is required and must exist.")
            return

        model_type = self.detect_model_type(model_path)
        if model_type not in self.strategies:
            logger.error(f"Unsupported model type: {model_type}")
            return

        strategy = self.strategies[model_type]

        try:
            # Instead of implementing our own inference pipeline, use Kohya's proven scripts
            inference_script = self._get_inference_script(model_type)
            
            # Build command for Kohya's inference script
            cmd = [
                sys.executable,  # Use current Python executable
                inference_script,
                "--ckpt", model_path,
                "--prompt", prompt,
                "--outdir", output_dir,
                "--W", str(width),
                "--H", str(height),
                "--sampler", "ddim",  # Default sampler
                "--steps", str(steps),
                "--scale", str(cfg_scale)
            ]
            
            # Add LoRA if specified
            if lora_path:
                cmd.extend(["--network_weights", lora_path])
                if lora_strength != 1.0:
                    cmd.extend(["--network_mul", str(lora_strength)])
            
            logger.info(f"Running Kohya inference: {' '.join(cmd)}")
            
            # Execute the inference script
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.sd_scripts_dir)
            
            if result.returncode == 0:
                logger.info("✅ Inference completed successfully")
                logger.info(result.stdout)
            else:
                logger.error("❌ Inference failed")
                logger.error(result.stderr)
                
        except Exception as e:
            logger.error(f"An error occurred during image generation: {e}")
            if model_path in self.models_cache:
                del self.models_cache[model_path]

    def _get_inference_script(self, model_type: str) -> str:
        """Get the appropriate Kohya inference script for model type"""
        # Mapping of model types to Kohya inference scripts
        script_mapping = {
            'sd15': 'gen_img.py',
            'sd20': 'gen_img.py', 
            'sdxl': 'sdxl_gen_img.py',
            'flux': 'flux_minimal_inference.py',
            'sd3': 'sd3_minimal_inference.py'
        }
        
        script_name = script_mapping.get(model_type, 'gen_img_diffusers.py')  # Default to diffusers script
        script_path = os.path.join(self.sd_scripts_dir, script_name)
        
        if not os.path.exists(script_path):
            # Fallback to general diffusers script
            fallback_path = os.path.join(self.sd_scripts_dir, 'gen_img_diffusers.py')
            if os.path.exists(fallback_path):
                logger.warning(f"Script {script_path} not found, using fallback: {fallback_path}")
                return fallback_path
            else:
                raise FileNotFoundError(f"No inference script found for {model_type}")
                
        return script_path

# Example usage:
if __name__ == '__main__':
    manager = KohyaInferenceManager()
    # This is a placeholder for a real model path
    sd15_model = '/path/to/your/sd15.safetensors'
    sdxl_model = '/path/to/your/sdxl.safetensors'

    if os.path.exists(sd15_model):
        manager.generate_image({
            'model_path': sd15_model,
            'prompt': 'a beautiful landscape painting'
        })

    if os.path.exists(sdxl_model):
        manager.generate_image({
            'model_path': sdxl_model,
            'prompt': 'an astronaut riding a horse on mars'
        })
