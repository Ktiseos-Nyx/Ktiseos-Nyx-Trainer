# core/flux_inference_manager.py
"""
Flux Inference Manager - Specialized for Flux.1 models
Complete implementation for Flux inference with LoRA support
"""
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional


class Flux_InferenceManager:
    """
    Specialized inference manager for Flux.1 models (dev/schnell).
    Handles model conversion, LoRA loading, and Flux-specific inference.
    """

    def __init__(self):
        self.project_root = os.getcwd()
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.sd_scripts_dir = os.path.join(self.trainer_dir, "derrian_backend", "sd_scripts")
        self.converted_models_dir = os.path.join(self.project_root, "converted_models", "flux")

        # Ensure directories exist
        os.makedirs(self.converted_models_dir, exist_ok=True)

        # Flux specific configuration
        self.model_type = 'flux'
        self.inference_script = 'flux_minimal_inference.py'
        self.inference_fallback = 'gen_img_diffusers.py'  # General diffusers script
        self.conversion_script = 'convert_diffusers_to_original_stable_diffusion.py'  # May work for Flux
        self.default_resolution = (1024, 1024)
        self.supports_guidance_scale = False  # Flux typically uses guidance_scale=1.0

    def _find_inference_script(self):
        """Find the Flux inference script with fallback"""
        # Try Flux-specific script first
        script_path = os.path.join(self.sd_scripts_dir, self.inference_script)
        if os.path.exists(script_path):
            return script_path

        # Try general diffusers script as fallback
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
        Convert Flux checkpoint to Diffusers format if needed.
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
                print(f"✅ Using cached Flux Diffusers model: {converted_dir}")
                return converted_dir

        # For Flux, often used directly from HuggingFace format
        if "black-forest-labs/FLUX" in model_path or "fal/AuraFlow" in model_path:
            print(f"📦 Using Flux model directly from HuggingFace: {model_path}")
            return model_path

        # Convert if not cached and is a checkpoint file
        if model_path.endswith(('.safetensors', '.ckpt')):
            print("🔄 Converting Flux checkpoint to Diffusers format...")
            conversion_script = self._find_conversion_script()

            if not conversion_script:
                print("⚠️ Flux conversion script not found, using original path")
                return model_path

            # Get Python executable
            from core.managers import (get_subprocess_environment,
                                       get_venv_python_path)
            python_executable = get_venv_python_path(self.sd_scripts_dir)
            if not os.path.exists(python_executable):
                python_executable = sys.executable

            env = get_subprocess_environment(self.project_root)

            try:
                # Attempt conversion (may not work for all Flux variants)
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
                    timeout=900  # Flux conversion may take longer
                )

                if result.returncode == 0:
                    print(f"✅ Flux model converted successfully: {converted_dir}")
                    return converted_dir
                else:
                    print(f"⚠️ Flux conversion failed, using original: {result.stderr[:200]}")
                    return model_path

            except subprocess.TimeoutExpired:
                print("⚠️ Flux model conversion timed out, using original")
                return model_path
            except Exception as e:
                print(f"⚠️ Flux conversion error, using original: {e}")
                return model_path

        return model_path

    def _prepare_inference_command(self, config: Dict[str, Any]) -> List[str]:
        """Prepare the Flux inference command"""
        inference_script = self._find_inference_script()
        if not inference_script:
            raise FileNotFoundError("Flux inference script not found")

        from core.managers import get_venv_python_path
        python_executable = get_venv_python_path(self.sd_scripts_dir)
        if not os.path.exists(python_executable):
            python_executable = sys.executable

        # Base command
        cmd = [python_executable, inference_script]

        # Model path (convert first if needed)
        model_path = self.convert_model_to_diffusers(config['model_path'])
        cmd.extend(["--ckpt", model_path])

        # Output settings
        output_dir = config.get('output_dir', os.path.join(self.project_root, 'inference_output'))
        os.makedirs(output_dir, exist_ok=True)
        cmd.extend(["--outdir", output_dir])

        # Basic generation parameters
        cmd.extend(["--prompt", config['prompt']])

        if config.get('negative_prompt'):
            cmd.extend(["--n", config['negative_prompt']])

        cmd.extend(["--steps", str(config.get('steps', 20))])
        cmd.extend(["--sampler", config.get('sampler', 'euler')])

        # Flux-specific: guidance scale is usually 1.0
        guidance_scale = config.get('guidance_scale', 1.0)
        cmd.extend(["--scale", str(guidance_scale)])

        # Resolution
        width = config.get('width', 1024)
        height = config.get('height', 1024)
        cmd.extend(["--W", str(width), "--H", str(height)])

        # Number of images
        batch_size = config.get('batch_size', 1)
        cmd.extend(["--batch_size", str(batch_size)])

        # Seed
        if config.get('seed'):
            cmd.extend(["--seed", str(config['seed'])])

        # LoRA support
        if config.get('lora_path'):
            cmd.extend(["--network_weights", config['lora_path']])
            if config.get('lora_strength'):
                cmd.extend(["--network_mul", str(config['lora_strength'])])

        # Precision settings
        precision = config.get('precision', 'bf16')
        if precision == 'fp16':
            cmd.append("--fp16")
        elif precision == 'bf16':
            cmd.append("--bf16")

        return cmd

    def run_inference(self, config: Dict[str, Any]) -> bool:
        """
        Run Flux inference with the specified configuration.
        """
        print("🎨 Starting Flux inference...")

        try:
            # Validate required parameters
            required_params = ['prompt', 'model_path']
            for param in required_params:
                if param not in config:
                    print(f"❌ Missing required parameter: {param}")
                    return False

            # Prepare command
            cmd = self._prepare_inference_command(config)
            print(f"🔧 Flux inference command: {' '.join(cmd[-10:])}...")  # Show last 10 args

            # Set up environment
            from core.managers import get_subprocess_environment
            env = get_subprocess_environment(self.project_root)

            # Execute inference
            original_cwd = os.getcwd()
            os.chdir(self.sd_scripts_dir)

            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                print("✅ Flux inference completed successfully!")
                print(f"📸 Images saved to: {config.get('output_dir', 'inference_output')}")
                return True
            else:
                print(f"❌ Flux inference failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("❌ Flux inference timed out")
            return False
        except Exception as e:
            print(f"❌ Flux inference error: {e}")
            return False
        finally:
            os.chdir(original_cwd)

    def generate_images(self, prompt: str, model_path: str, lora_path: Optional[str] = None,
                       num_images: int = 1, steps: int = 20, guidance_scale: float = 1.0,
                       width: int = 1024, height: int = 1024, seed: Optional[int] = None,
                       negative_prompt: Optional[str] = None, lora_strength: float = 1.0) -> bool:
        """
        Generate images using Flux with optional LoRA.
        Note: Flux typically uses guidance_scale=1.0 for best results.
        """
        config = {
            'prompt': prompt,
            'model_path': model_path,
            'steps': steps,
            'guidance_scale': guidance_scale,
            'width': width,
            'height': height,
            'batch_size': num_images,
            'precision': 'bf16'  # Flux prefers bf16
        }

        if lora_path:
            config['lora_path'] = lora_path
            config['lora_strength'] = lora_strength

        if seed is not None:
            config['seed'] = seed

        if negative_prompt:
            config['negative_prompt'] = negative_prompt

        return self.run_inference(config)

    def validate_flux_config(self, config: Dict[str, Any]) -> bool:
        """Validate Flux-specific inference configuration"""
        warnings = []

        # Check guidance scale
        guidance_scale = config.get('guidance_scale', 1.0)
        if guidance_scale != 1.0:
            warnings.append(f"💡 Flux works best with guidance_scale=1.0 (current: {guidance_scale})")

        # Check resolution
        width = config.get('width', 1024)
        height = config.get('height', 1024)
        if width % 64 != 0 or height % 64 != 0:
            warnings.append("⚠️ Resolution should be divisible by 64 for Flux")

        # Check steps
        steps = config.get('steps', 20)
        if steps > 50:
            warnings.append(f"💡 Flux converges quickly, {steps} steps may be excessive")

        for warning in warnings:
            print(warning)

        return len(warnings) == 0
