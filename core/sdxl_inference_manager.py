# core/sdxl_inference_manager.py
"""
SDXL Inference Manager - Specialized for SDXL inference
Handles model conversion and inference execution with SDXL optimizations.
"""
import os
import subprocess
import sys
from typing import Any, Dict, Optional


class SDXL_InferenceManager:
    """
    Specialized inference manager for SDXL models.
    Handles Diffusers conversion and SDXL-specific inference.
    """

    def __init__(self):
        self.project_root = os.getcwd()
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.sd_scripts_dir = os.path.join(self.trainer_dir, "derrian_backend", "sd_scripts")
        self.converted_models_dir = os.path.join(self.project_root, "converted_models", "sdxl")

        # Ensure directories exist
        os.makedirs(self.converted_models_dir, exist_ok=True)

        # SDXL specific configuration
        self.model_type = 'sdxl'
        self.inference_script = 'sdxl_gen_img.py'
        self.inference_fallback = 'sdxl_minimal_inference.py'
        self.conversion_script = 'custom_tools/convert_sdxl_checkpoint_to_diffusers.py'
        self.default_resolution = (1024, 1024)

    def _find_inference_script(self):
        """Find the SDXL inference script with fallback"""
        # Try primary script first
        script_path = os.path.join(self.sd_scripts_dir, self.inference_script)
        if os.path.exists(script_path):
            return script_path

        # Try fallback
        fallback_path = os.path.join(self.sd_scripts_dir, self.inference_fallback)
        if os.path.exists(fallback_path):
            return fallback_path

        return None

    def _find_conversion_script(self):
        """Find the model conversion script"""
        script_path = os.path.join(self.sd_scripts_dir, self.conversion_script)
        if os.path.exists(script_path):
            return script_path
        return None

    def convert_model_to_diffusers(self, model_path: str) -> str:
        """
        Convert SDXL checkpoint to Diffusers format if needed.
        Returns path to converted model directory.
        """
        import hashlib

        # Generate unique directory name based on model path
        model_hash = hashlib.md5(model_path.encode()).hexdigest()[:8]
        model_name = os.path.basename(model_path).replace('.safetensors', '').replace('.ckpt', '')
        converted_dir = os.path.join(self.converted_models_dir, f"{model_name}_{model_hash}")

        # Check if already converted
        if os.path.exists(converted_dir) and os.path.isdir(converted_dir):
            # Verify it's a valid diffusers directory
            if os.path.exists(os.path.join(converted_dir, "model_index.json")):
                print(f"✅ Using cached SDXL Diffusers model: {converted_dir}")
                return converted_dir

        # Convert if not cached
        print("🔄 Converting SDXL model to Diffusers format...")
        conversion_script = self._find_conversion_script()

        if not conversion_script:
            print("❌ Model conversion script not found")
            return model_path

        # Get Python executable
        from core.managers import (get_subprocess_environment,
                                   get_venv_python_path)
        python_executable = get_venv_python_path(self.sd_scripts_dir)
        if not os.path.exists(python_executable):
            python_executable = sys.executable

        env = get_subprocess_environment(self.project_root)

        try:
            # Run SDXL conversion (may need different parameters)
            conversion_cmd = [
                python_executable, conversion_script,
                "--checkpoint_path", model_path,
                "--dump_path", converted_dir,
                "--is_sdxl"  # SDXL-specific flag
            ]

            result = subprocess.run(
                conversion_cmd,
                cwd=self.sd_scripts_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=600  # SDXL conversion takes longer
            )

            if result.returncode == 0:
                print(f"✅ SDXL model converted successfully: {converted_dir}")
                return converted_dir
            else:
                print(f"❌ SDXL conversion failed: {result.stderr}")
                return model_path

        except subprocess.TimeoutExpired:
            print("❌ SDXL model conversion timed out")
            return model_path
        except Exception as e:
            print(f"❌ SDXL conversion error: {e}")
            return model_path

    def run_inference(self, config: Dict[str, Any]) -> bool:
        """
        Run SDXL inference with the specified configuration.
        """
        # Implementation placeholder
        print("🎨 SDXL inference not yet implemented")
        return False

    def generate_images(self, prompt: str, model_path: str, lora_path: Optional[str] = None,
                       num_images: int = 1, steps: int = 20, cfg_scale: float = 7.0,
                       width: int = 1024, height: int = 1024) -> bool:
        """
        Generate images using SDXL with optional LoRA.
        """
        # Implementation placeholder
        print("🎨 SDXL image generation not yet implemented")
        return False
