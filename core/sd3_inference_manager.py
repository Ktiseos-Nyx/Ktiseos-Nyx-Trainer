# core/sd3_inference_manager.py
"""
SD3 Inference Manager - Specialized for Stable Diffusion 3 models
Complete implementation for SD3 inference with LoRA support
"""
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional


class SD3_InferenceManager:
    """
    Specialized inference manager for SD3 models (Medium/Large).
    Handles model conversion, LoRA loading, and SD3-specific inference.
    """

    def __init__(self):
        self.project_root = os.getcwd()
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.sd_scripts_dir = os.path.join(self.trainer_dir, "derrian_backend", "sd_scripts")
        self.converted_models_dir = os.path.join(self.project_root, "converted_models", "sd3")

        # Ensure directories exist
        os.makedirs(self.converted_models_dir, exist_ok=True)

        # SD3 specific configuration
        self.model_type = 'sd3'
        self.inference_script = 'sd3_minimal_inference.py'
        self.inference_fallback = 'gen_img_diffusers.py'  # General diffusers script
        self.conversion_script = 'convert_sd3_to_diffusers.py'  # SD3-specific if available
        self.conversion_fallback = 'convert_stable_diffusion_to_diffusers.py'  # General fallback
        self.default_resolution = (1024, 1024)
        self.supports_guidance_scale = True  # SD3 uses standard guidance scaling

    def _find_inference_script(self):
        """Find the SD3 inference script with fallback"""
        # Try SD3-specific script first
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
        # Try SD3-specific conversion first
        script_path = os.path.join(self.sd_scripts_dir, self.conversion_script)
        if os.path.exists(script_path):
            return script_path

        # Try general conversion as fallback
        fallback_path = os.path.join(self.sd_scripts_dir, self.conversion_fallback)
        if os.path.exists(fallback_path):
            return fallback_path

        return None

    def convert_model_to_diffusers(self, model_path: str) -> str:
        """
        Convert SD3 checkpoint to Diffusers format if needed.
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
                print(f"✅ Using cached SD3 Diffusers model: {converted_dir}")
                return converted_dir

        # For SD3, often used directly from HuggingFace format
        if "stabilityai/stable-diffusion-3" in model_path:
            print(f"📦 Using SD3 model directly from HuggingFace: {model_path}")
            return model_path

        # Convert if not cached and is a checkpoint file
        if model_path.endswith(('.safetensors', '.ckpt')):
            print("🔄 Converting SD3 checkpoint to Diffusers format...")
            conversion_script = self._find_conversion_script()

            if not conversion_script:
                print("⚠️ SD3 conversion script not found, using original path")
                return model_path

            # Get Python executable
            from core.managers import (get_subprocess_environment,
                                       get_venv_python_path)
            python_executable = get_venv_python_path(self.sd_scripts_dir)
            if not os.path.exists(python_executable):
                python_executable = sys.executable

            env = get_subprocess_environment(self.project_root)

            try:
                # Attempt SD3 conversion
                conversion_cmd = [
                    python_executable, conversion_script,
                    "--checkpoint_path", model_path,
                    "--dump_path", converted_dir
                ]

                # Add SD3-specific flags if using specialized script
                if "sd3" in os.path.basename(conversion_script):
                    conversion_cmd.append("--is_sd3")

                result = subprocess.run(
                    conversion_cmd,
                    cwd=self.sd_scripts_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=900  # SD3 conversion may take time
                )

                if result.returncode == 0:
                    print(f"✅ SD3 model converted successfully: {converted_dir}")
                    return converted_dir
                else:
                    print(f"⚠️ SD3 conversion failed, using original: {result.stderr[:200]}")
                    return model_path

            except subprocess.TimeoutExpired:
                print("⚠️ SD3 model conversion timed out, using original")
                return model_path
            except Exception as e:
                print(f"⚠️ SD3 conversion error, using original: {e}")
                return model_path

        return model_path

    def _prepare_inference_command(self, config: Dict[str, Any]) -> List[str]:
        """Prepare the SD3 inference command"""
        inference_script = self._find_inference_script()
        if not inference_script:
            raise FileNotFoundError("SD3 inference script not found")

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
        cmd.extend(["--sampler", config.get('sampler', 'euler_a')])  # SD3 often uses euler_a

        # SD3 guidance scale (typically higher than Flux)
        guidance_scale = config.get('guidance_scale', 7.0)
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

        # Precision settings (SD3 prefers bf16)
        precision = config.get('precision', 'bf16')
        if precision == 'fp16':
            cmd.append("--fp16")
        elif precision == 'bf16':
            cmd.append("--bf16")

        return cmd

    def run_inference(self, config: Dict[str, Any]) -> bool:
        """
        Run SD3 inference with the specified configuration.
        """
        print("🎨 Starting SD3 inference...")

        try:
            # Validate required parameters
            required_params = ['prompt', 'model_path']
            for param in required_params:
                if param not in config:
                    print(f"❌ Missing required parameter: {param}")
                    return False

            # SD3-specific warnings
            if not self._validate_sd3_inference_config(config):
                print("⚠️ SD3 inference validation warnings (proceeding anyway)")

            # Prepare command
            cmd = self._prepare_inference_command(config)
            print(f"🔧 SD3 inference command: {' '.join(cmd[-10:])}...")  # Show last 10 args

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
                timeout=900  # SD3 may take longer than other models
            )

            if result.returncode == 0:
                print("✅ SD3 inference completed successfully!")
                print(f"📸 Images saved to: {config.get('output_dir', 'inference_output')}")
                return True
            else:
                print(f"❌ SD3 inference failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("❌ SD3 inference timed out")
            return False
        except Exception as e:
            print(f"❌ SD3 inference error: {e}")
            return False
        finally:
            os.chdir(original_cwd)

    def _validate_sd3_inference_config(self, config: Dict[str, Any]) -> bool:
        """Validate SD3-specific inference configuration"""
        warnings = []

        # Check batch size (SD3 is memory-intensive)
        batch_size = config.get('batch_size', 1)
        if batch_size > 2:
            warnings.append(f"⚠️ SD3 is VRAM-intensive, batch_size={batch_size} may cause OOM")

        # Check resolution
        width = config.get('width', 1024)
        height = config.get('height', 1024)
        if width % 64 != 0 or height % 64 != 0:
            warnings.append("⚠️ Resolution should be divisible by 64 for SD3")

        if width < 512 or height < 512:
            warnings.append("⚠️ SD3 works best with resolution ≥ 512x512")

        # Check guidance scale
        guidance_scale = config.get('guidance_scale', 7.0)
        if guidance_scale < 3.0 or guidance_scale > 15.0:
            warnings.append(f"💡 SD3 guidance_scale typically 3-15 (current: {guidance_scale})")

        # Memory warning
        warnings.append("💡 SD3 requires significant VRAM (16GB+ recommended for inference)")

        for warning in warnings:
            print(warning)

        return len(warnings) <= 1  # Allow the memory warning

    def generate_images(self, prompt: str, model_path: str, lora_path: Optional[str] = None,
                       num_images: int = 1, steps: int = 20, guidance_scale: float = 7.0,
                       width: int = 1024, height: int = 1024, seed: Optional[int] = None,
                       negative_prompt: Optional[str] = None, lora_strength: float = 1.0,
                       sampler: str = 'euler_a') -> bool:
        """
        Generate images using SD3 with optional LoRA.
        SD3 uses standard guidance scaling (unlike Flux).
        """
        config = {
            'prompt': prompt,
            'model_path': model_path,
            'steps': steps,
            'guidance_scale': guidance_scale,
            'width': width,
            'height': height,
            'batch_size': num_images,
            'sampler': sampler,
            'precision': 'bf16'  # SD3 prefers bf16
        }

        if lora_path:
            config['lora_path'] = lora_path
            config['lora_strength'] = lora_strength

        if seed is not None:
            config['seed'] = seed

        if negative_prompt:
            config['negative_prompt'] = negative_prompt

        return self.run_inference(config)
