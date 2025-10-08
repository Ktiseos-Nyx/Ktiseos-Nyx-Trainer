# core/sd15_inference_manager.py
"""
SD 1.5 Inference Manager - Specialized for SD 1.5/2.x inference
Handles model conversion and inference execution.
"""
import os
import subprocess
import sys
from typing import Any, Dict, Optional


class SD15_InferenceManager:
    """
    Specialized inference manager for SD 1.5/2.x models.
    Handles Diffusers conversion and inference execution.
    """

    def __init__(self):
        self.project_root = os.getcwd()
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.sd_scripts_dir = os.path.join(self.trainer_dir, "derrian_backend", "sd_scripts")
        self.converted_models_dir = os.path.join(self.project_root, "converted_models", "sd15")

        # Ensure directories exist
        os.makedirs(self.converted_models_dir, exist_ok=True)

        # SD 1.5 specific configuration
        self.model_type = 'sd15'
        self.inference_script = 'gen_img_diffusers.py'
        self.conversion_script = 'tools/convert_diffusers20_original_sd.py'
        self.default_resolution = (512, 512)

    def _find_inference_script(self):
        """Find the SD 1.5 inference script"""
        script_path = os.path.join(self.sd_scripts_dir, self.inference_script)
        if os.path.exists(script_path):
            return script_path
        return None

    def _find_conversion_script(self):
        """Find the model conversion script"""
        script_path = os.path.join(self.sd_scripts_dir, self.conversion_script)
        if os.path.exists(script_path):
            return script_path
        return None

    def convert_model_to_diffusers(self, model_path: str) -> str:
        """
        Convert SD 1.5 checkpoint to Diffusers format if needed.
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
                print(f"✅ Using cached SD 1.5 Diffusers model: {converted_dir}")
                return converted_dir

        # Convert if not cached
        print("🔄 Converting SD 1.5 model to Diffusers format...")
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
            # Run conversion
            conversion_cmd = [
                python_executable, conversion_script,
                "--checkpoint_path", model_path,
                "--dump_path", converted_dir
            ]

            result = subprocess.run(
                conversion_cmd,
                cwd=self.sd_scripts_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                print(f"✅ SD 1.5 model converted successfully: {converted_dir}")
                return converted_dir
            else:
                print(f"❌ Conversion failed: {result.stderr}")
                return model_path

        except subprocess.TimeoutExpired:
            print("❌ Model conversion timed out")
            return model_path
        except Exception as e:
            print(f"❌ Conversion error: {e}")
            return model_path

    def run_inference(self, config: Dict[str, Any]) -> bool:
        """
        Run SD 1.5 inference with the specified configuration.
        """
        # Implementation placeholder
        print("🎨 SD 1.5 inference not yet implemented")
        return False

    def generate_images(self, prompt: str, model_path: str, lora_path: Optional[str] = None,
                       num_images: int = 1, steps: int = 20, cfg_scale: float = 7.0) -> bool:
        """
        Generate images using SD 1.5 with optional LoRA.
        """
        # Implementation placeholder
        print("🎨 SD 1.5 image generation not yet implemented")
        return False
