# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# core/managers.py
import datetime
import os
import platform
import re
import shutil
import subprocess
import sys
from urllib.parse import urlparse


def detect_python_environment():
    """Detect what type of Python environment we're running in"""
    current_python = sys.executable

    # Check for conda
    if 'conda' in current_python.lower() or 'anaconda' in current_python.lower():
        return 'conda'

    # Check for virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        if 'venv' in current_python:
            return 'venv'
        elif 'env' in current_python:
            return 'virtualenv'
        else:
            return 'virtual_env'  # Generic virtual environment

    # Check for pyenv
    if 'pyenv' in current_python:
        return 'pyenv'

    # Check for system install locations
    if sys.platform == "win32":
        if 'Program Files' in current_python or 'AppData' in current_python:
            return 'system'
    else:
        if current_python.startswith(('/usr/bin', '/usr/local/bin', '/bin')):
            return 'system'

    return 'unknown'

def get_venv_python_path(base_dir):
    """Get cross-platform virtual environment Python path with flexible detection"""

    # Try multiple common environment structures in order of preference
    python_candidates = []

    if sys.platform == "win32":
        # Windows environments
        python_candidates = [
            os.path.join(base_dir, "venv", "Scripts", "python.exe"),  # Standard venv
            os.path.join(base_dir, "env", "Scripts", "python.exe"),   # Alternative venv name
            os.path.join(base_dir, ".venv", "Scripts", "python.exe"), # Hidden venv
            os.path.join(base_dir, "conda", "python.exe"),            # Conda in subdir
            os.path.join(base_dir, "python.exe"),                     # Direct python
        ]
    else:
        # Unix/Linux/Mac environments
        python_candidates = [
            os.path.join(base_dir, "venv", "bin", "python"),          # Standard venv
            os.path.join(base_dir, "venv", "bin", "python3"),         # Python3 explicit
            os.path.join(base_dir, "env", "bin", "python"),           # Alternative venv name
            os.path.join(base_dir, "env", "bin", "python3"),          # Alt venv python3
            os.path.join(base_dir, ".venv", "bin", "python"),         # Hidden venv
            os.path.join(base_dir, ".venv", "bin", "python3"),        # Hidden venv python3
            os.path.join(base_dir, "conda", "bin", "python"),         # Conda in subdir
            os.path.join(base_dir, "conda", "bin", "python3"),        # Conda python3
            os.path.join(base_dir, "python"),                         # Direct python
            os.path.join(base_dir, "python3"),                        # Direct python3
        ]

    # Try each candidate and return first existing executable
    for candidate in python_candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    # If no virtual environment found, fall back to current Python
    # This handles conda environments, pyenv, system installs, etc.
    current_env = detect_python_environment()
    if current_env != 'system':
        # We're already in a managed environment, use it directly
        return sys.executable

    # Last resort: use current Python executable
    return sys.executable

def get_subprocess_environment(project_root=None):
    """
    Create standardized environment for subprocess calls with proper PYTHONPATH and CUDA setup.
    This prevents the CAME optimizer import errors and other module not found issues.
    """
    if project_root is None:
        project_root = os.getcwd()

    # Start with current environment
    env = os.environ.copy()

    # Memory optimization
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    env['CUDA_LAUNCH_BLOCKING'] = '1'

    # Setup PYTHONPATH for custom optimizers (CAME, etc.)
    derrian_dir = os.path.join(project_root, "trainer", "derrian_backend")
    custom_scheduler_dir = os.path.join(derrian_dir, "custom_scheduler")

    if os.path.exists(custom_scheduler_dir):
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{custom_scheduler_dir}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = custom_scheduler_dir

    # Setup CUDA/CuDNN library paths for Linux
    if sys.platform.startswith('linux'):
        cuda_path = env.get("CUDA_PATH", "/usr/local/cuda")
        new_ld_library_path = f"{cuda_path}/lib64:{cuda_path}/extras/CUPTI/lib64"

        # Add common CuDNN paths - these ARE supposed to be hardcoded system paths!
        cudnn_paths = [
            f"{cuda_path}/lib",
            f"{cuda_path}/targets/x86_64-linux/lib",
            "/usr/lib/x86_64-linux-gnu",
        ]
        for p in cudnn_paths:
            if os.path.exists(p):
                new_ld_library_path += f":{p}"

        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{new_ld_library_path}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = new_ld_library_path

    # ROCm support for AMD GPUs
    if not env.get('HSA_OVERRIDE_GFX_VERSION'):
        env['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

    return env

class SetupManager:
    def __init__(self):
        # Light initialization - just paths, no detection spam!
        self.project_root = os.getcwd()
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.derrian_dir = os.path.join(self.trainer_dir, "derrian_backend")
        self.sd_scripts_dir = os.path.join(self.derrian_dir, "sd_scripts")
        self.lycoris_dir = os.path.join(self.derrian_dir, "lycoris")

        # Lazy-loaded properties
        self._environment_detected = False
        self._is_vastai = None
        self._correct_venv_path = None

        # Submodule URLs - Only need derrian_backend which includes sd_scripts and lycoris
        self.submodules = {
            'derrian_backend': {
                'url': 'https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend.git',
                'path': self.derrian_dir,
                'description': 'Derrian\'s backend with Kohya scripts, LyCORIS, and custom optimizers'
            }
        }

    def _detect_environment(self):
        """Detect environment only when actually needed - not on every widget load!"""
        if self._environment_detected:
            return

        # Check for VastAI first
        self._is_vastai = bool(os.environ.get('VAST_CONTAINERLABEL') or '/workspace' in self.project_root)
        if self._is_vastai:
            print("🐳 VastAI detected - using current environment instead of hardcoded paths")

        # Always detect current Python/pip environment (conda, venv, etc.) - never hardcode paths
        # This follows CLAUDE.md requirement: NEVER hardcode paths or environment assumptions
        self._correct_venv_path = self._detect_current_pip()

        self._environment_detected = True

    def _detect_current_pip(self):
        """Detect the pip executable for the current Python environment"""
        # Use our new flexible detection system
        python_path = get_venv_python_path(os.path.dirname(sys.executable))
        env_type = detect_python_environment()

        # Convert python path to pip path for direct pip execution
        pip_candidates = []

        if sys.platform == "win32":
            # Windows pip locations
            if python_path.endswith('python.exe'):
                direct_pip = python_path.replace('python.exe', 'pip.exe')
            else:
                direct_pip = os.path.join(os.path.dirname(python_path), 'pip.exe')
            pip_candidates.append(direct_pip)
        else:
            # Unix pip locations
            if python_path.endswith(('python', 'python3')):
                direct_pip = python_path.replace('python3', 'pip3').replace('python', 'pip')
            else:
                direct_pip = os.path.join(os.path.dirname(python_path), 'pip')
            pip_candidates.extend([
                direct_pip,
                direct_pip + '3',  # pip3 variant
            ])

        # Try direct pip first, fall back to python -m pip
        for pip_path in pip_candidates:
            if os.path.isfile(pip_path) and os.access(pip_path, os.X_OK):
                print(f"🐍 {env_type.title()} environment detected - using pip: {pip_path}")
                return pip_path

        # Fallback to python -m pip (most reliable)
        print(f"🐍 {env_type.title()} environment detected - using python -m pip")
        return f"{python_path} -m pip"

    def validate_and_fix_deep_dependencies(self):
        """Comprehensive dependency validation and auto-fixing"""
        print("🔍 Running comprehensive dependency validation...")
        print("=" * 60)

        all_good = True

        # 1. CUDA/cuDNN Compatibility Check
        print("🎮 Checking CUDA/cuDNN compatibility...")
        cuda_status = self._check_cuda_cudnn_versions()
        if not cuda_status['compatible']:
            print(f"⚠️ CUDA issue detected: {cuda_status['issue']}")
            if self._fix_cuda_cudnn_mismatch(cuda_status):
                print("✅ CUDA issue fixed!")
            else:
                print("❌ CUDA issue persists - manual intervention needed")
                all_good = False
        else:
            print("✅ CUDA/cuDNN compatibility looks good")

        # 2. PyTorch CUDA Compatibility
        print("\n🔥 Checking PyTorch CUDA compatibility...")
        pytorch_status = self._check_pytorch_cuda_compatibility()
        if not pytorch_status['compatible']:
            print(f"⚠️ PyTorch issue: {pytorch_status['issue']}")
            if self._reinstall_correct_pytorch(pytorch_status):
                print("✅ PyTorch issue fixed!")
            else:
                print("❌ PyTorch issue persists")
                all_good = False
        else:
            print("✅ PyTorch CUDA compatibility verified")

        # 3. Derrian Backend Dependencies
        print("\n🚀 Checking Derrian backend dependencies...")
        backend_status = self._check_derrian_backend_deps()
        if not backend_status['complete']:
            print(f"⚠️ Backend issue: {backend_status['issue']}")
            if self._fix_derrian_backend_install(backend_status):
                print("✅ Backend dependencies fixed!")
            else:
                print("❌ Backend issue persists")
                all_good = False
        else:
            print("✅ Derrian backend dependencies complete")

        print("\n" + "=" * 60)
        if all_good:
            print("🎉 All dependencies validated and ready for training!")
        else:
            print("⚠️ Some issues remain - check logs above")

        return all_good

    def validate_deep_dependencies_readonly(self):
        """Comprehensive dependency validation (READ-ONLY - no fixes or reinstalls)"""
        print("🔍 Running comprehensive dependency diagnostics (read-only mode)...")
        print("=" * 60)

        all_good = True

        # 1. CUDA/cuDNN Compatibility Check (READ-ONLY)
        print("🎮 Checking CUDA/cuDNN compatibility...")
        cuda_status = self._check_cuda_cudnn_versions()
        if not cuda_status['compatible']:
            print(f"❌ CUDA issue detected: {cuda_status['issue']}")
            print("🔧 Primary fix: Run installer.py to fix CUDA/PyTorch compatibility")

            # Give specific manual fix suggestions
            gpu_type = cuda_status.get('gpu_type', 'nvidia')
            if gpu_type == 'nvidia':
                print("🛠️ Manual fix alternatives:")
                print("   • Reinstall PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
                print("   • Check CUDA version: nvidia-smi")
                print("   • Update GPU drivers from NVIDIA website")
            elif gpu_type == 'amd':
                print("🛠️ Manual fix alternatives:")
                print("   • Install ROCm PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2")
                print("   • Or try ZLUDA: Follow ZLUDA installation guide for NVIDIA CUDA on AMD")
                print("   • Or use DirectML: pip install torch-directml")
            all_good = False
        else:
            print("✅ CUDA/cuDNN compatibility looks good!")

        # 2. PyTorch CUDA Compatibility Check (READ-ONLY)
        print("\n🔥 Checking PyTorch CUDA compatibility...")
        pytorch_status = self._check_pytorch_cuda_compatibility()
        if not pytorch_status['compatible']:
            print(f"❌ PyTorch issue: {pytorch_status['issue']}")
            print("🔧 Primary fix: Run installer.py to reinstall correct PyTorch version")
            print("🛠️ Manual fix alternatives:")
            print("   • Check PyTorch version: python -c \"import torch; print(torch.__version__)\"")
            print("   • Check CUDA availability: python -c \"import torch; print(torch.cuda.is_available())\"")
            print("   • Uninstall and reinstall: pip uninstall torch torchvision && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            all_good = False
        else:
            print("✅ PyTorch CUDA compatibility verified!")

        # 3. Derrian Backend Dependencies (READ-ONLY)
        print("\n🚀 Checking Derrian backend dependencies...")
        backend_status = self._check_derrian_backend_deps()
        if not backend_status['complete']:
            print(f"❌ Backend issue: {backend_status['issue']}")
            print("🔧 Primary fix: Run installer.py to install missing backend dependencies")
            print("🛠️ Manual fix alternatives:")
            print("   • Clone submodules: git submodule update --init --recursive")
            print("   • Install missing packages: pip install transformers accelerate diffusers")
            print("   • Check trainer directory: ls -la trainer/derrian_backend/")
            all_good = False
        else:
            print("✅ Derrian backend dependencies complete!")

        # 4. Additional Tools Check (READ-ONLY)
        print("\n🛠️ Checking additional ML tools...")
        tools_to_check = {
            'transformers': 'pip install transformers',
            'accelerate': 'pip install accelerate',
            'diffusers': 'pip install diffusers',
            'xformers': 'pip install xformers'
        }

        missing_tools = []
        for tool, install_cmd in tools_to_check.items():
            try:
                __import__(tool)
                print(f"✅ {tool} available")
            except ImportError:
                print(f"❌ {tool} not available")
                missing_tools.append((tool, install_cmd))
                all_good = False

        if missing_tools:
            print("🔧 Primary fix: Run installer.py to install missing tools")
            print("🛠️ Manual fix alternatives:")
            for tool, cmd in missing_tools:
                print(f"   • Install {tool}: {cmd}")

        # 5. Common Issues Troubleshooting
        if not all_good:
            print("\n🚨 Common Troubleshooting Steps:")
            print("   • Clear pip cache: pip cache purge")
            print("   • Update pip: pip install --upgrade pip")
            print("   • Restart Jupyter kernel after installing packages")
            print("   • Check available disk space: df -h")
            print("   • Check Python environment: which python && python --version")

        print("\n" + "=" * 60)
        if all_good:
            print("🎉 All diagnostic checks passed!")
            print("💚 Your environment appears to be fully configured.")
            print("🚀 You should be ready for LoRA training!")
        else:
            print("⚠️ Issues detected in your environment")
            print()
            print("🔧 RECOMMENDED FIXES:")
            print("   1. Run installer.py (automatic fix for most issues)")
            print("   2. Try the manual commands listed above if installer.py doesn't work")
            print("   3. Restart Jupyter kernel after installing packages")
            print("   4. Run this diagnostic again to verify fixes")
            print()
            print("📋 Note: This was a READ-ONLY diagnostic - no changes were made")
            print("💡 Copy-paste the commands above if you need to fix things manually")

        return all_good

    def _detect_gpu_type(self):
        """Detect GPU type and acceleration method"""
        gpu_info = {
            'type': 'unknown',
            'devices': [],
            'acceleration': None,
            'rocm_available': False,
            'zluda_available': False,
            'directml_available': False
        }

        try:
            import torch

            # Check for AMD GPU via ROCm
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                gpu_info['type'] = 'amd'
                gpu_info['acceleration'] = 'rocm'
                gpu_info['rocm_available'] = True
                if torch.cuda.is_available():  # ROCm uses cuda interface
                    gpu_info['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                return gpu_info

            # Check for NVIDIA GPU
            if torch.cuda.is_available():
                try:
                    device_name = torch.cuda.get_device_name(0).lower()
                    if 'nvidia' in device_name or 'geforce' in device_name or 'rtx' in device_name or 'gtx' in device_name:
                        gpu_info['type'] = 'nvidia'
                        gpu_info['acceleration'] = 'cuda'
                        gpu_info['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                        return gpu_info
                    elif 'amd' in device_name or 'radeon' in device_name:
                        # AMD GPU detected via CUDA interface (ZLUDA likely)
                        gpu_info['type'] = 'amd'
                        gpu_info['acceleration'] = 'zluda'
                        gpu_info['zluda_available'] = True
                        gpu_info['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                        return gpu_info
                except Exception:
                    pass

            # Check for DirectML (Windows AMD fallback)
            if sys.platform == "win32":
                try:
                    import torch_directml
                    gpu_info['acceleration'] = 'directml'
                    gpu_info['directml_available'] = True
                    if torch_directml.is_available():
                        # Don't assume AMD - DirectML works on AMD, Intel, and even some NVIDIA
                        # Try to detect actual GPU vendor
                        actual_gpu_type = self._detect_directml_gpu_vendor()
                        gpu_info['type'] = actual_gpu_type
                        gpu_info['devices'] = [f'DirectML Device ({actual_gpu_type.upper()})']
                        return gpu_info
                except ImportError:
                    pass

            # Check system for AMD GPUs via other methods
            gpu_info.update(self._detect_amd_gpu_system())

        except ImportError:
            gpu_info['issue'] = 'PyTorch not installed'

        return gpu_info

    def _detect_amd_gpu_system(self):
        """Detect AMD GPU via system calls"""
        amd_info = {}

        if sys.platform == "win32":
            # Windows: Check via wmic
            try:
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                                      capture_output=True, text=True, timeout=5)
                if 'AMD' in result.stdout or 'Radeon' in result.stdout:
                    amd_info['type'] = 'amd'
                    amd_info['devices'] = ['AMD GPU (detected via system)']
            except Exception:
                pass
        else:
            # Linux: Check via lspci
            try:
                result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True, timeout=5)
                if 'AMD' in result.stdout or 'ATI' in result.stdout:
                    amd_info['type'] = 'amd'
                    # Extract GPU names
                    lines = result.stdout.split('\n')
                    gpu_lines = [line for line in lines if ('AMD' in line or 'ATI' in line) and ('VGA' in line or 'Display' in line)]
                    amd_info['devices'] = gpu_lines[:3]  # Limit output
            except Exception:
                pass

        return amd_info

    def _detect_directml_gpu_vendor(self) -> str:
        """Detect actual GPU vendor when using DirectML using proper GPU utilities"""

        # 1. Check for NVIDIA first using nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and 'GPU' in result.stdout:
                return 'nvidia'
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # 2. Check for AMD using rocm-smi
        try:
            result = subprocess.run(['rocm-smi', '--showid'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return 'amd'
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # 3. Fallback AMD check using radeontop (if available)
        try:
            result = subprocess.run(['radeontop', '-d', '-l1'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return 'amd'
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # 4. Intel detection - only for ARM-based systems (like Surface Pro X, M1 Macs running Windows ARM)
        if platform.machine().lower() in ['arm64', 'aarch64']:
            try:
                # Check for Intel Arc on ARM systems (rare but possible)
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and any(intel_indicator in result.stdout.lower()
                                                for intel_indicator in ['intel arc', 'intel iris']):
                    return 'intel'
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        # 5. Final fallback - we know DirectML is available but can't determine vendor
        print("⚠️ DirectML available but GPU vendor unknown - this may affect training optimization")
        return 'unknown'

    def _check_cuda_cudnn_versions(self):
        """Check CUDA and cuDNN version compatibility (NVIDIA) or ROCm compatibility (AMD)"""
        gpu_info = self._detect_gpu_type()

        if gpu_info['type'] == 'amd':
            return self._check_amd_compatibility(gpu_info)
        elif gpu_info['type'] == 'nvidia':
            return self._check_nvidia_cuda_compatibility()
        else:
            return {
                'compatible': False,
                'issue': 'No supported GPU detected',
                'gpu_type': 'unknown',
                'fix_type': 'install_gpu_support'
            }

    def _check_nvidia_cuda_compatibility(self):
        """Check NVIDIA CUDA compatibility"""
        try:
            import torch

            if not torch.cuda.is_available():
                return {
                    'compatible': False,
                    'issue': 'CUDA not available in PyTorch',
                    'cuda_version': None,
                    'cudnn_version': None,
                    'gpu_type': 'nvidia',
                    'fix_type': 'reinstall_pytorch_cuda'
                }

            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version()

            # Check for known problematic combinations
            # Note: cuDNN 9.x works fine with PyTorch 2.1+, only flag if PyTorch is old
            if cudnn_version and str(cudnn_version).startswith('9'):
                try:
                    import torch
                    pytorch_version = torch.__version__
                    # Only flag as incompatible if PyTorch is old (< 2.1)
                    major, minor = map(int, pytorch_version.split('.')[:2])
                    if major < 2 or (major == 2 and minor < 1):
                        return {
                            'compatible': False,
                            'issue': f'cuDNN {cudnn_version} incompatible with PyTorch {pytorch_version} (need PyTorch 2.1+)',
                            'cuda_version': cuda_version,
                            'cudnn_version': cudnn_version,
                            'gpu_type': 'nvidia',
                            'fix_type': 'upgrade_pytorch'
                        }
                    else:
                        # cuDNN 9.x + PyTorch 2.1+ = perfectly fine!
                        print(f"ℹ️ cuDNN {cudnn_version} + PyTorch {pytorch_version} = Compatible ✅")
                except Exception:
                    # If we can't check PyTorch version, assume it's fine
                    pass

            return {
                'compatible': True,
                'cuda_version': cuda_version,
                'cudnn_version': cudnn_version,
                'gpu_type': 'nvidia'
            }

        except ImportError:
            return {
                'compatible': False,
                'issue': 'PyTorch not installed',
                'gpu_type': 'nvidia',
                'fix_type': 'install_pytorch_cuda'
            }

    def _check_amd_compatibility(self, gpu_info):
        """Check AMD GPU compatibility with available acceleration methods"""
        acceleration = gpu_info.get('acceleration')

        if acceleration == 'rocm':
            return self._check_rocm_compatibility(gpu_info)
        elif acceleration == 'zluda':
            return self._check_zluda_compatibility(gpu_info)
        elif acceleration == 'directml':
            return self._check_directml_compatibility(gpu_info)
        else:
            # No acceleration detected, suggest options
            return {
                'compatible': False,
                'issue': 'AMD GPU detected but no acceleration method available',
                'gpu_type': 'amd',
                'devices': gpu_info.get('devices', []),
                'fix_type': 'install_amd_support',
                'suggestions': self._get_amd_support_suggestions()
            }

    def _check_rocm_compatibility(self, gpu_info):
        """Check ROCm compatibility"""
        try:
            import torch

            if not torch.cuda.is_available():
                return {
                    'compatible': False,
                    'issue': 'ROCm PyTorch installed but GPU not accessible',
                    'gpu_type': 'amd',
                    'acceleration': 'rocm',
                    'fix_type': 'fix_rocm_setup'
                }

            # Test ROCm functionality
            try:
                test_tensor = torch.randn(10, device='cuda')
                hip_version = torch.version.hip if hasattr(torch.version, 'hip') else 'Unknown'

                return {
                    'compatible': True,
                    'gpu_type': 'amd',
                    'acceleration': 'rocm',
                    'hip_version': hip_version,
                    'devices': gpu_info.get('devices', [])
                }
            except Exception as e:
                return {
                    'compatible': False,
                    'issue': f'ROCm test failed: {str(e)}',
                    'gpu_type': 'amd',
                    'acceleration': 'rocm',
                    'fix_type': 'fix_rocm_setup'
                }

        except ImportError:
            return {
                'compatible': False,
                'issue': 'ROCm PyTorch not installed',
                'gpu_type': 'amd',
                'fix_type': 'install_rocm_pytorch'
            }

    def _check_zluda_compatibility(self, gpu_info):
        """Check ZLUDA compatibility"""
        try:
            import torch

            if not torch.cuda.is_available():
                return {
                    'compatible': False,
                    'issue': 'ZLUDA detected but CUDA interface not working',
                    'gpu_type': 'amd',
                    'acceleration': 'zluda',
                    'fix_type': 'fix_zluda_setup'
                }

            # Test ZLUDA functionality
            try:
                test_tensor = torch.randn(10, device='cuda')

                return {
                    'compatible': True,
                    'gpu_type': 'amd',
                    'acceleration': 'zluda',
                    'cuda_version': torch.version.cuda,
                    'devices': gpu_info.get('devices', []),
                    'note': 'ZLUDA experimental - performance may vary'
                }
            except Exception as e:
                return {
                    'compatible': False,
                    'issue': f'ZLUDA test failed: {str(e)}',
                    'gpu_type': 'amd',
                    'acceleration': 'zluda',
                    'fix_type': 'fix_zluda_setup'
                }

        except ImportError:
            return {
                'compatible': False,
                'issue': 'PyTorch not installed for ZLUDA',
                'gpu_type': 'amd',
                'fix_type': 'install_pytorch_zluda'
            }

    def _check_directml_compatibility(self, gpu_info):
        """Check DirectML compatibility"""
        try:
            import torch_directml

            if not torch_directml.is_available():
                return {
                    'compatible': False,
                    'issue': 'DirectML installed but not available',
                    'gpu_type': 'amd',
                    'acceleration': 'directml',
                    'fix_type': 'fix_directml_setup'
                }

            return {
                'compatible': True,
                'gpu_type': 'amd',
                'acceleration': 'directml',
                'devices': gpu_info.get('devices', []),
                'note': 'DirectML may have limited LoRA training support'
            }

        except ImportError:
            return {
                'compatible': False,
                'issue': 'DirectML not installed',
                'gpu_type': 'amd',
                'fix_type': 'install_directml'
            }

    def _get_amd_support_suggestions(self):
        """Get platform-specific AMD support suggestions"""
        suggestions = []

        if sys.platform.startswith('linux'):
            suggestions.extend([
                'Install ROCm PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0',
                'Try ZLUDA: Download from https://github.com/vosen/ZLUDA',
                'Ensure ROCm drivers installed: AMD ROCm 6.1+'
            ])
        elif sys.platform == 'win32':
            suggestions.extend([
                'Try ZLUDA: Download from https://github.com/vosen/ZLUDA',
                'Install DirectML: pip install torch-directml',
                'Note: ROCm not supported on Windows'
            ])
        else:
            suggestions.append('AMD GPU support limited on this platform')

        return suggestions

    def _check_pytorch_cuda_compatibility(self):
        """Check if PyTorch installation matches GPU type (CUDA for NVIDIA, ROCm for AMD)"""
        gpu_info = self._detect_gpu_type()

        if gpu_info['type'] == 'amd':
            return self._check_amd_pytorch_compatibility(gpu_info)
        elif gpu_info['type'] == 'nvidia':
            return self._check_nvidia_pytorch_compatibility()
        else:
            return {
                'compatible': False,
                'issue': 'No supported GPU detected',
                'fix_type': 'install_gpu_support'
            }

    def _check_nvidia_pytorch_compatibility(self):
        """Check NVIDIA PyTorch compatibility"""
        try:
            import torch

            if not torch.cuda.is_available():
                return {
                    'compatible': False,
                    'issue': 'PyTorch CUDA not available',
                    'torch_cuda': None,
                    'gpu_type': 'nvidia',
                    'fix_type': 'reinstall_pytorch_cuda'
                }

            torch_cuda = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"

            # Basic sanity check
            if device_count == 0:
                return {
                    'compatible': False,
                    'issue': 'No CUDA devices detected',
                    'gpu_type': 'nvidia',
                    'fix_type': 'check_cuda_install'
                }

            # Test basic CUDA operations
            try:
                test_tensor = torch.randn(10, device='cuda')
                _ = test_tensor * 2.0
            except Exception as e:
                return {
                    'compatible': False,
                    'issue': f'CUDA operation failed: {e}',
                    'gpu_type': 'nvidia',
                    'fix_type': 'reinstall_pytorch_cuda'
                }

            return {
                'compatible': True,
                'torch_cuda': torch_cuda,
                'device_count': device_count,
                'device_name': device_name,
                'gpu_type': 'nvidia'
            }

        except ImportError:
            return {
                'compatible': False,
                'issue': 'PyTorch not installed',
                'gpu_type': 'nvidia',
                'fix_type': 'install_pytorch'
            }

    def _check_amd_pytorch_compatibility(self, gpu_info):
        """Check AMD PyTorch compatibility"""
        acceleration = gpu_info.get('acceleration')

        if acceleration == 'rocm':
            return self._check_amd_rocm_pytorch()
        elif acceleration == 'zluda':
            return self._check_amd_zluda_pytorch()
        elif acceleration == 'directml':
            return self._check_amd_directml_pytorch()
        else:
            return {
                'compatible': False,
                'issue': 'AMD GPU detected but no PyTorch acceleration available',
                'gpu_type': 'amd',
                'fix_type': 'install_amd_pytorch',
                'suggestions': self._get_amd_support_suggestions()
            }

    def _check_amd_rocm_pytorch(self):
        """Check AMD ROCm PyTorch compatibility"""
        try:
            import torch

            # Check for ROCm build
            if not hasattr(torch.version, 'hip') or torch.version.hip is None:
                return {
                    'compatible': False,
                    'issue': 'AMD GPU detected but PyTorch CUDA build installed (need ROCm build)',
                    'gpu_type': 'amd',
                    'acceleration': 'rocm',
                    'fix_type': 'install_rocm_pytorch'
                }

            if not torch.cuda.is_available():
                return {
                    'compatible': False,
                    'issue': 'ROCm PyTorch installed but GPU not accessible',
                    'gpu_type': 'amd',
                    'acceleration': 'rocm',
                    'hip_version': torch.version.hip,
                    'fix_type': 'fix_rocm_setup'
                }

            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"

            # Test ROCm functionality
            try:
                test_tensor = torch.randn(10, device='cuda')
                _ = test_tensor * 2.0

                return {
                    'compatible': True,
                    'gpu_type': 'amd',
                    'acceleration': 'rocm',
                    'hip_version': torch.version.hip,
                    'device_count': device_count,
                    'device_name': device_name
                }
            except Exception as e:
                return {
                    'compatible': False,
                    'issue': f'ROCm test failed: {str(e)}',
                    'gpu_type': 'amd',
                    'acceleration': 'rocm',
                    'fix_type': 'fix_rocm_setup'
                }

        except ImportError:
            return {
                'compatible': False,
                'issue': 'ROCm PyTorch not installed',
                'gpu_type': 'amd',
                'fix_type': 'install_rocm_pytorch'
            }

    def _check_amd_zluda_pytorch(self):
        """Check AMD ZLUDA PyTorch compatibility"""
        try:
            import torch

            if not torch.cuda.is_available():
                return {
                    'compatible': False,
                    'issue': 'ZLUDA setup detected but CUDA interface not working',
                    'gpu_type': 'amd',
                    'acceleration': 'zluda',
                    'fix_type': 'fix_zluda_setup'
                }

            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"

            # Test ZLUDA functionality
            try:
                test_tensor = torch.randn(10, device='cuda')
                _ = test_tensor * 2.0

                return {
                    'compatible': True,
                    'gpu_type': 'amd',
                    'acceleration': 'zluda',
                    'torch_cuda': torch.version.cuda,
                    'device_count': device_count,
                    'device_name': device_name,
                    'note': 'ZLUDA experimental - some features may not work'
                }
            except Exception as e:
                return {
                    'compatible': False,
                    'issue': f'ZLUDA test failed: {str(e)}',
                    'gpu_type': 'amd',
                    'acceleration': 'zluda',
                    'fix_type': 'fix_zluda_setup'
                }

        except ImportError:
            return {
                'compatible': False,
                'issue': 'PyTorch not installed for ZLUDA',
                'gpu_type': 'amd',
                'fix_type': 'install_pytorch_zluda'
            }

    def _check_amd_directml_pytorch(self):
        """Check AMD DirectML PyTorch compatibility"""
        try:
            import torch_directml

            if not torch_directml.is_available():
                return {
                    'compatible': False,
                    'issue': 'DirectML installed but not available',
                    'gpu_type': 'amd',
                    'acceleration': 'directml',
                    'fix_type': 'fix_directml_setup'
                }

            return {
                'compatible': True,
                'gpu_type': 'amd',
                'acceleration': 'directml',
                'device_count': 1,  # DirectML typically shows as single device
                'device_name': 'DirectML Device',
                'note': 'DirectML has limited LoRA training support'
            }

        except ImportError:
            return {
                'compatible': False,
                'issue': 'DirectML not installed',
                'gpu_type': 'amd',
                'fix_type': 'install_directml'
            }

    def _check_derrian_backend_deps(self):
        """Check if Derrian backend and CAME optimizers are properly installed"""
        issues = []

        # Check if sd_scripts directory exists
        if not os.path.exists(self.sd_scripts_dir):
            issues.append("sd_scripts directory missing")

        # Check for key training scripts
        key_scripts = ['train_network.py', 'sdxl_train_network.py']
        for script in key_scripts:
            if not os.path.exists(os.path.join(self.sd_scripts_dir, script)):
                issues.append(f"Missing training script: {script}")

        # Check for CAME optimizer - try multiple import paths
        came_available = False

        # Method 1: Direct import (if installed globally)
        try:
            import LoraEasyCustomOptimizer.came
            came_available = True
        except ImportError:
            pass

        # Method 2: Try from custom_scheduler directory
        if not came_available:
            sys_path_backup = sys.path.copy()
            try:
                custom_scheduler_dir = os.path.join(self.derrian_dir, "custom_scheduler")
                if os.path.exists(custom_scheduler_dir):
                    sys.path.insert(0, custom_scheduler_dir)
                    import LoraEasyCustomOptimizer.came
                    came_available = True
            except ImportError:
                pass
            finally:
                sys.path = sys_path_backup

        # Method 3: Check if it's available as 'came' directly
        if not came_available:
            try:
                from custom_scheduler.LoraEasyCustomOptimizer.came import CAME
                # If we can import the CAME class, it's available
                came_available = True
            except ImportError:
                pass

        # Method 4: Try subprocess test (most reliable)
        if not came_available:
            try:
                venv_python = get_venv_python_path(self.project_root)
                env = get_subprocess_environment(self.project_root)
                result = subprocess.run(
                    [venv_python, '-c', 'from custom_scheduler.LoraEasyCustomOptimizer.came import CAME; print("CAME available")'],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=5
                )
                if result.returncode == 0:
                    came_available = True
            except Exception:
                pass

        if not came_available:
            issues.append("CAME optimizer not available")

        # Check for LyCORIS
        lycoris_available = False
        lycoris_dir = os.path.join(self.sd_scripts_dir, "networks")
        if os.path.exists(lycoris_dir):
            lycoris_files = ['lora.py', 'dylora.py']  # Basic check
            if any(os.path.exists(os.path.join(lycoris_dir, f)) for f in lycoris_files):
                lycoris_available = True

        if not lycoris_available:
            issues.append("LyCORIS networks not available")

        return {
            'complete': len(issues) == 0,
            'issues': issues,
            'issue': '; '.join(issues) if issues else None
        }

    def _fix_cuda_cudnn_mismatch(self, status):
        """Attempt to fix GPU compatibility issues (NVIDIA CUDA or AMD)"""
        print("🔧 Attempting to fix GPU compatibility issues...")

        fix_type = status.get('fix_type')
        gpu_type = status.get('gpu_type', 'nvidia')

        # NVIDIA GPU fixes
        if gpu_type == 'nvidia':
            if fix_type in ['reinstall_pytorch_cuda', 'reinstall_pytorch_cuda8']:
                return self._reinstall_pytorch_with_cuda8()
            elif fix_type == 'install_pytorch':
                return self._install_pytorch_with_cuda()

        # AMD GPU fixes
        elif gpu_type == 'amd':
            if fix_type == 'install_rocm_pytorch':
                return self._install_rocm_pytorch()
            elif fix_type == 'install_pytorch_zluda':
                return self._install_pytorch_zluda()
            elif fix_type == 'install_directml':
                return self._install_directml()
            elif fix_type == 'install_amd_support':
                return self._install_amd_support()
            elif fix_type == 'install_amd_pytorch':
                return self._install_amd_support()
            elif fix_type == 'fix_rocm_setup':
                return self._fix_rocm_setup()
            elif fix_type == 'fix_zluda_setup':
                return self._fix_zluda_setup()
            elif fix_type == 'fix_directml_setup':
                return self._fix_directml_setup()

        # Generic GPU support
        elif fix_type == 'install_gpu_support':
            print("❓ No supported GPU detected. Please check:")
            print("  • NVIDIA GPU: Install NVIDIA drivers + CUDA")
            print("  • AMD GPU (Linux): Install ROCm drivers")
            print("  • AMD GPU (Windows): Install AMD drivers")
            return False

        print(f"❌ Unknown fix type: {fix_type} for {gpu_type} GPU")
        return False

    def _reinstall_correct_pytorch(self, status):
        """Reinstall PyTorch with correct CUDA support"""
        return self._reinstall_pytorch_with_cuda8()

    def _reinstall_pytorch_with_cuda8(self):
        """Install PyTorch with cuDNN 8.x compatibility"""
        print("📦 Installing PyTorch with cuDNN 8.x compatibility...")

        try:
            # Uninstall existing PyTorch
            pip_cmd = self.correct_venv_path
            if isinstance(pip_cmd, list):
                uninstall_cmd = pip_cmd + ['uninstall', '-y', 'torch', 'torchvision', 'torchaudio']
            else:
                uninstall_cmd = [pip_cmd, 'uninstall', '-y', 'torch', 'torchvision', 'torchaudio']

            subprocess.run(uninstall_cmd, check=True, capture_output=True)

            # Install PyTorch with CUDA 11.8 (compatible with cuDNN 8.x)
            if isinstance(pip_cmd, list):
                install_cmd = pip_cmd + [
                    'install', 'torch', 'torchvision', 'torchaudio',
                    '--index-url', 'https://download.pytorch.org/whl/cu118'
                ]
            else:
                install_cmd = [pip_cmd, 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118']

            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            print("✅ PyTorch with CUDA 11.8 installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to reinstall PyTorch: {e}")
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
            return False

    def _install_pytorch_with_cuda(self):
        """Install PyTorch with CUDA from scratch"""
        return self._reinstall_pytorch_with_cuda8()

    # AMD GPU Support Methods
    def _install_rocm_pytorch(self):
        """Install ROCm PyTorch for AMD GPUs"""
        print("🔥 Installing ROCm PyTorch for AMD GPU support...")

        venv_python = get_venv_python_path(self.project_root)

        if sys.platform.startswith('linux'):
            commands = [
                # Uninstall existing PyTorch first
                [venv_python, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"],
                # Install ROCm PyTorch
                [venv_python, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
                 "--index-url", "https://download.pytorch.org/whl/rocm6.0"]
            ]

            for cmd in commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if result.returncode != 0:
                        print(f"⚠️ Command failed: {' '.join(cmd)}")
                        print(f"Error: {result.stderr}")
                        return False
                except subprocess.TimeoutExpired:
                    print("⚠️ Installation timed out")
                    return False

            print("✅ ROCm PyTorch installation completed")
            return True
        else:
            print("❌ ROCm PyTorch only supported on Linux")
            return False

    def _install_pytorch_zluda(self):
        """Install PyTorch for ZLUDA compatibility"""
        print("🔥 Installing PyTorch for ZLUDA compatibility...")

        # ZLUDA uses regular CUDA PyTorch - the magic happens in the ZLUDA runtime
        return self._reinstall_pytorch_with_cuda8()

    def _install_directml(self):
        """Install DirectML for AMD GPU support on Windows"""
        print("🔥 Installing DirectML for AMD GPU support...")

        if sys.platform != "win32":
            print("❌ DirectML only supported on Windows")
            return False

        venv_python = get_venv_python_path(self.project_root)

        commands = [
            # Install DirectML
            [venv_python, "-m", "pip", "install", "torch-directml"]
        ]

        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"⚠️ Command failed: {' '.join(cmd)}")
                    print(f"Error: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                print("⚠️ Installation timed out")
                return False

        print("✅ DirectML installation completed")
        return True

    def _install_amd_support(self):
        """Install appropriate AMD GPU support based on platform"""
        print("🔥 Installing AMD GPU support...")

        if sys.platform.startswith('linux'):
            print("🐧 Linux detected - installing ROCm PyTorch...")
            return self._install_rocm_pytorch()
        elif sys.platform == 'win32':
            print("🪟 Windows detected - installing DirectML...")
            return self._install_directml()
        else:
            print("❌ AMD GPU support not available on this platform")
            return False

    def _fix_rocm_setup(self):
        """Fix ROCm setup issues"""
        print("🔧 Fixing ROCm setup issues...")

        # Check for common ROCm environment issues
        fixes_applied = []

        # Set HSA_OVERRIDE_GFX_VERSION if needed
        if not os.environ.get('HSA_OVERRIDE_GFX_VERSION'):
            os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
            fixes_applied.append('Set HSA_OVERRIDE_GFX_VERSION=10.3.0')

        # Try reinstalling ROCm PyTorch
        if self._install_rocm_pytorch():
            fixes_applied.append('Reinstalled ROCm PyTorch')

        if fixes_applied:
            print(f"✅ Applied fixes: {', '.join(fixes_applied)}")
            return True
        else:
            print("❌ Unable to fix ROCm setup")
            return False

    def _fix_zluda_setup(self):
        """Fix ZLUDA setup issues"""
        print("🔧 Fixing ZLUDA setup issues...")
        print("ℹ️ ZLUDA requires manual setup - please ensure:")
        print("  1. ZLUDA runtime libraries are in PATH")
        print("  2. CUDA PyTorch is installed")
        print("  3. AMD drivers support compute")
        print("📖 See: https://github.com/vosen/ZLUDA")

        # Try reinstalling regular PyTorch for ZLUDA
        if self._reinstall_pytorch_with_cuda8():
            print("✅ Reinstalled PyTorch for ZLUDA compatibility")
            return True
        else:
            print("❌ Unable to fix ZLUDA setup")
            return False

    def _fix_directml_setup(self):
        """Fix DirectML setup issues"""
        print("🔧 Fixing DirectML setup issues...")

        # Try reinstalling DirectML
        if self._install_directml():
            return True
        else:
            print("❌ Unable to fix DirectML setup")
            return False

    def _fix_derrian_backend_install(self, status):
        """Fix Derrian backend installation issues"""
        print("🔧 Attempting to fix Derrian backend issues...")

        issues = status.get('issues', [])
        success = True

        # Fix missing sd_scripts
        if any('sd_scripts' in issue for issue in issues):
            print("📦 Fixing sd_scripts installation...")
            if not self._install_sd_scripts():
                success = False

        # Fix CAME optimizer
        if any('CAME' in issue for issue in issues):
            print("📦 Fixing CAME optimizer installation...")
            if not self._install_came_optimizer():
                success = False

        return success

    def _install_sd_scripts(self):
        """Ensure sd_scripts is properly installed"""
        try:
            # This should trigger the existing setup logic
            return self.setup_environment()
        except Exception as e:
            print(f"❌ Failed to install sd_scripts: {e}")
            return False

    def _install_came_optimizer(self):
        """Ensure CAME optimizer is properly installed using standardized subprocess environment"""
        try:
            custom_scheduler_dir = os.path.join(self.derrian_dir, "custom_scheduler")

            # Check if already installed
            try:
                venv_python = get_venv_python_path(self.project_root)
                env = get_subprocess_environment(self.project_root)

                # Test if CAME optimizer is already available
                result = subprocess.run(
                    [venv_python, '-c', 'from custom_scheduler.LoraEasyCustomOptimizer.came import CAME; print("CAME available")'],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=10
                )
                if result.returncode == 0:
                    print("✅ CAME optimizer already installed")
                    return True
            except Exception:
                pass  # Not installed, continue

            if os.path.exists(custom_scheduler_dir):
                print(f"📦 Installing CAME optimizer from {custom_scheduler_dir}...")

                # Get proper venv python and environment
                venv_python = get_venv_python_path(self.project_root)
                if not os.path.exists(venv_python):
                    print(f"⚠️ Virtual environment python not found at {venv_python}")
                    # Try current Python executable as fallback
                    venv_python = sys.executable
                    print(f"🔄 Using current Python executable: {venv_python}")

                # Get standardized subprocess environment (fixes CAME import issues!)
                env = get_subprocess_environment(self.project_root)

                # Use pip install -e for editable installation
                result = subprocess.run(
                    [venv_python, '-m', 'pip', 'install', '-e', '.'],
                    cwd=custom_scheduler_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env
                )
                print("✅ CAME optimizer installed successfully")
                print(f"   Output: {result.stdout}")
                return True
            else:
                print("❌ CAME optimizer directory not found at custom_scheduler/")
                return False

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install CAME optimizer: {e}")
            if e.stdout:
                print(f"   Stdout: {e.stdout}")
            if e.stderr:
                print(f"   Stderr: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error installing CAME optimizer: {e}")
            return False

    @property
    def is_vastai(self):
        """Lazy property - only detects environment when asked"""
        if not self._environment_detected:
            self._detect_environment()
        return self._is_vastai

    @property
    def correct_venv_path(self):
        """Lazy property - only detects environment when asked"""
        if not self._environment_detected:
            self._detect_environment()
        return self._correct_venv_path

    def check_setup_status(self):
        """Holy shit dude you forgot to set up go back up mode! 🚨"""
        if not os.path.exists(self.derrian_dir):
            print("🚨 SETUP MISSING: trainer/derrian_backend not found!")
            print("💡 Go back and run the setup widget first!")
            return False
        if not os.path.exists(self.sd_scripts_dir):
            print("🚨 SETUP INCOMPLETE: sd_scripts submodule missing!")
            print("💡 Re-run setup to initialize submodules!")
            return False
        return True

    def _check_and_install_packages(self):
        packages = ["git", "aria2c"]
        for pkg in packages:
            if not shutil.which(pkg):
                print(f"{pkg} not found. Attempting to install...")
                try:
                    subprocess.run(["apt-get", "update"], check=True)
                    subprocess.run(["apt-get", "install", "-y", pkg], check=True)
                    print(f"{pkg} installed successfully.")
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"Failed to install {pkg}. Please install it manually. Error: {e}")
                    return False
        return True

    def _setup_submodule(self, name, config):
        """Clone and setup a single submodule"""
        path = config['path']
        url = config['url']

        if os.path.exists(path) and os.listdir(path):
            print(f"✅ {config['description']} already exists")
            return True

        print(f"📥 Cloning {config['description']}...")
        print(f"   URL: {url}")

        try:
            # Remove if exists but empty
            if os.path.exists(path):
                shutil.rmtree(path)

            # Clone the repository
            subprocess.run(["git", "clone", url, path], check=True)

            # Pin to specific commit if specified
            if 'commit' in config:
                print(f"📌 Pinning to commit: {config['commit'][:8]}...")
                subprocess.run(["git", "checkout", config['commit']], cwd=path, check=True)

            # Special handling for derrian_backend - initialize its nested submodules
            if name == 'derrian_backend':
                print("🔗 Initializing nested submodules (sd_scripts, lycoris)...")
                try:
                    # Initialize and update nested submodules
                    subprocess.run(["git", "submodule", "init"], cwd=path, check=True)
                    subprocess.run(["git", "submodule", "update"], cwd=path, check=True)
                    print("✅ Nested submodules initialized")

                    # 🐍 PYTHON MODULE FIX: Add missing __init__.py files for CAME optimizer
                    # Python requires __init__.py files to recognize directories as modules
                    print("🔧 Fixing Python module detection for custom_scheduler...")
                    custom_scheduler_dir = os.path.join(path, "custom_scheduler")
                    lora_optimizer_dir = os.path.join(custom_scheduler_dir, "LoraEasyCustomOptimizer")

                    # Create __init__.py files if directories exist but files are missing
                    init_files_created = []
                    if os.path.exists(custom_scheduler_dir):
                        init_file = os.path.join(custom_scheduler_dir, "__init__.py")
                        if not os.path.exists(init_file):
                            with open(init_file, 'w') as f:
                                f.write("# Auto-generated __init__.py for Python module detection\n")
                            init_files_created.append("custom_scheduler/__init__.py")

                    if os.path.exists(lora_optimizer_dir):
                        init_file = os.path.join(lora_optimizer_dir, "__init__.py")
                        if not os.path.exists(init_file):
                            with open(init_file, 'w') as f:
                                f.write("# Auto-generated __init__.py for Python module detection\n")
                            init_files_created.append("custom_scheduler/LoraEasyCustomOptimizer/__init__.py")

                    if init_files_created:
                        print(f"✅ Created missing __init__.py files: {', '.join(init_files_created)}")
                        print("🎯 CAME optimizer should now be detectable by Python!")
                    else:
                        print("ℹ️ Python module files already exist or directories not found")

                    # Run Derrian's installer to set up dependencies (non-interactive mode)
                    print("📦 Running Derrian's installer for dependencies...")
                    installer_path = os.path.join(path, "installer.py")
                    if os.path.exists(installer_path):
                        try:
                            # Use 'local' argument to skip interactive prompts
                            subprocess.run([sys.executable, installer_path, "local"], cwd=path, check=True)
                            print("✅ Derrian's dependencies installed")
                        except subprocess.CalledProcessError as e:
                            print(f"⚠️ Derrian's installer failed: {e}")
                            print("💡 Will continue with manual dependency installation")
                    else:
                        print("⚠️ Derrian's installer not found, skipping dependency setup")

                    # Install custom scheduler optimizers (CAME, REX, etc.)
                    print("📦 Installing custom scheduler optimizers...")
                    custom_scheduler_path = os.path.join(path, "custom_scheduler")
                    setup_py_path = os.path.join(custom_scheduler_path, "setup.py")
                    if os.path.exists(setup_py_path):
                        try:
                            # Proper setup.py install command
                            subprocess.run([sys.executable, "setup.py", "install"], cwd=custom_scheduler_path, check=True)
                            print("✅ Custom scheduler optimizers (LoraEasyCustomOptimizer) installed")
                        except subprocess.CalledProcessError as e:
                            print(f"⚠️ Custom scheduler installation failed: {e}")
                            print("💡 Optimizers exist but may need manual installation")
                    else:
                        print("⚠️ Custom scheduler setup.py not found")

                    # Run PyTorch fix for Windows users (just in case)
                    print("🔧 Running PyTorch fix (Windows compatibility)...")
                    fix_torch_path = os.path.join(path, "fix_torch.py")
                    if os.path.exists(fix_torch_path):
                        try:
                            subprocess.run([sys.executable, fix_torch_path], cwd=path, check=True)
                            print("✅ PyTorch Windows compatibility fix applied")
                        except subprocess.CalledProcessError:
                            print("⚠️ PyTorch fix failed (likely not needed on this system)")
                    else:
                        print("⚠️ PyTorch fix script not found")

                except subprocess.CalledProcessError as e:
                    print(f"⚠️ Nested submodule setup failed: {e}")
                    print("💡 You may need to run setup again or manually initialize submodules")

            print(f"✅ {config['description']} setup complete")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to setup {name}: {e}")
            return False

    def _setup_all_submodules(self):
        """Setup all required submodules"""
        print("🏗️ Setting up training backend submodules...")

        # Ensure trainer directory exists
        os.makedirs(self.trainer_dir, exist_ok=True)

        success_count = 0
        for name, config in self.submodules.items():
            if self._setup_submodule(name, config):
                success_count += 1

        if success_count == len(self.submodules):
            print(f"🎉 All {len(self.submodules)} submodules setup successfully!")
            return True
        else:
            print(f"⚠️ Only {success_count}/{len(self.submodules)} submodules setup successfully")
            return False

    def _setup_custom_optimizers(self):
        """Setup Derrian's custom optimizers with proper attribution"""
        print("🧪 Setting up custom optimizers (CAME, REX)...")

        custom_scheduler_dir = os.path.join(self.derrian_dir, "custom_scheduler")

        if not os.path.exists(custom_scheduler_dir):
            print("❌ Derrian's custom_scheduler directory not found")
            return False

        # Add to Python path for imports
        import sys
        if custom_scheduler_dir not in sys.path:
            sys.path.insert(0, custom_scheduler_dir)
            print(f"📂 Added to Python path: {custom_scheduler_dir}")

        # Find and install the custom optimizer package
        setup_py = None
        for root, dirs, files in os.walk(custom_scheduler_dir):
            if 'setup.py' in files:
                setup_py = root
                break

        if setup_py:
            print("📦 Installing Derrian's custom optimizers...")
            try:
                # Install in editable mode
                subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True, cwd=setup_py)
                print("✅ Custom optimizers installed successfully")

                # Test imports
                try:
                    import LoraEasyCustomOptimizer.came
                    import LoraEasyCustomOptimizer.RexAnnealingWarmRestarts
                    print("✅ Custom optimizers verified working")
                    return True
                except ImportError as e:
                    print(f"⚠️ Custom optimizers installed but import failed: {e}")
                    return False

            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install custom optimizers: {e}")
                return False
        else:
            print("❌ setup.py not found in custom_scheduler")
            return False


    def _get_python_version(self):
        """Get the current Python version (major.minor)"""
        try:
            import sys
            version = f"{sys.version_info.major}.{sys.version_info.minor}"
            return version
        except Exception as e:
            print(f"⚠️ Could not detect Python version: {e}")
            return "3.12"  # Default fallback

    def _is_environment_setup(self):
        """Checks if the core components of the training environment are already in place."""
        # Check 1: Is the backend repo cloned and looks valid?
        if not os.path.exists(self.trainer_dir):
            return False
        try:
            contents = os.listdir(self.trainer_dir)
            junk_files = ['.ipynb_checkpoints', '__pycache__', '.DS_Store', '.git']
            real_files = [f for f in contents if f not in junk_files and not f.startswith('.')]
            expected_files = ['custom_scheduler', 'main.py', 'requirements.txt', 'sd_scripts']
            has_derrian_files = any(f in real_files for f in expected_files)
            if not (len(real_files) > 0 and has_derrian_files):
                return False
        except Exception:
            return False

        # Check 2: Do the actual training scripts exist?
        possible_sd_locations = [
            os.path.join(self.trainer_dir, "sd_scripts"),
            os.path.join(self.trainer_dir, "sd-scripts"),
            self.trainer_dir
        ]
        sd_scripts_found = False
        for location in possible_sd_locations:
            if os.path.exists(location):
                try:
                    train_scripts = [f for f in os.listdir(location) if 'train_network.py' in f]
                    if train_scripts:
                        sd_scripts_found = True
                        break
                except Exception:
                    continue # Ignore errors from listing non-directories

        return sd_scripts_found

    def setup_environment(self):
        """
        The new, unified master setup method.
        Orchestrates the entire backend installation process.
        """
        print("Checking environment setup...")

        if not self._is_environment_setup():
            print("\n🚨 Environment not fully set up!")
            print("   Please run `python installer.py` in your terminal to complete the setup.")
            print("   This will install all necessary components and dependencies.")
            return False
        else:
            print("\n✅ Environment appears to be set up.")
            print("   Displaying PyTorch and GPU status:")
            gpu_info = self._detect_gpu_type()
            print(f"   GPU Type: {gpu_info['type'].upper()}")
            if gpu_info['devices']:
                for i, device in enumerate(gpu_info['devices']):
                    print(f"     Device {i}: {device}")
            else:
                print("     No specific GPU devices detected or listed.")
            print(f"   Acceleration: {gpu_info['acceleration'].upper() if gpu_info['acceleration'] else 'N/A'}")

            try:
                import torch
                print(f"   PyTorch Version: {torch.__version__}")
                print(f"   CUDA Available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"   CUDA Version: {torch.version.cuda}")
            except ImportError:
                print("   PyTorch not installed or not detectable.")

            print("\n🎉 Environment is ready for use!")
            return True

    def _apply_special_fixes_and_installs(self):
        """
        Applies necessary platform-specific patches and performs editable installs.
        This logic was consolidated from the old installer scripts.
        """
        print("🔧 Applying special fixes and performing editable installs...")
        all_success = True

        # Get the correct pip command for editable installs
        pip_cmd_str = self._detect_current_pip()
        if isinstance(pip_cmd_str, str) and " -m pip" in pip_cmd_str:
            pip_cmd = pip_cmd_str.split()
        else:
            pip_cmd = [pip_cmd_str]

        # --- Editable Installs ---
        editable_installs = {
            "LyCORIS": self.lycoris_dir,
            "Custom Optimizers": os.path.join(self.derrian_dir, "custom_scheduler"),
            "Kohya's SD Scripts": self.sd_scripts_dir
        }

        for name, path in editable_installs.items():
            if os.path.exists(os.path.join(path, 'setup.py')):
                print(f"  - Performing editable install for {name}...")
                try:
                    install_cmd = pip_cmd + ['install', '-e', '.']
                    subprocess.run(install_cmd, cwd=path, check=True, capture_output=True)
                    print(f"    ✅ {name} installed successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"    ❌ Failed to install {name} in editable mode: {e.stderr.decode('utf-8', errors='ignore')}")
                    all_success = False
            else:
                print(f"  - Skipping editable install for {name} (setup.py not found).")

        # --- Platform-Specific Fixes ---

        # 1. Windows bitsandbytes fix
        if sys.platform == "win32":
            print("  - Applying Windows-specific fix for bitsandbytes...")
            try:
                # This logic is based on the old derrian_backend/installer.py
                bnb_src_dir = os.path.join(self.sd_scripts_dir, 'bitsandbytes_windows')
                # Find site-packages directory
                result = subprocess.run([sys.executable, '-c', 'import site; print(site.getsitepackages()[0])'], capture_output=True, text=True, check=True)
                site_packages = result.stdout.strip()
                bnb_dest_dir = os.path.join(site_packages, 'bitsandbytes')

                if os.path.exists(bnb_dest_dir):
                    # TODO: Add logic to select the correct CUDA version DLL
                    # For now, we'll assume a common one. This can be improved later.
                    dll_to_copy = 'libbitsandbytes_cuda118.dll'
                    src_file = os.path.join(bnb_src_dir, dll_to_copy)
                    dest_file = os.path.join(bnb_dest_dir, 'libbitsandbytes_cudaall.dll') # The name it expects
                    if os.path.exists(src_file):
                        shutil.copy2(src_file, dest_file)
                        print(f"    ✅ Copied {dll_to_copy} to {dest_file}")
                    else:
                        print(f"    ⚠️ Could not find source DLL: {src_file}")
                        all_success = False
                else:
                    print("    ⚠️ bitsandbytes directory not found in site-packages. Cannot apply fix.")
                    all_success = False
            except Exception as e:
                print(f"    ❌ Error applying bitsandbytes fix: {e}")
                all_success = False

        # 2. PyTorch version file fix
        print("  - Checking if PyTorch version patch is needed...")
        try:
            import torch
            if torch.__version__ in ["2.0.0", "2.0.1"]:
                print(f"  - Applying patch for PyTorch {torch.__version__}...")
                fix_script_path = os.path.join(self.derrian_dir, 'fix_torch.py')
                if os.path.exists(fix_script_path):
                    subprocess.run([sys.executable, fix_script_path], check=True, capture_output=True)
                    print("    ✅ PyTorch patch applied successfully.")
                else:
                    print("    ⚠️ fix_torch.py script not found. Cannot apply patch.")
                    all_success = False
            else:
                print(f"  - PyTorch version is {torch.__version__}. No patch needed.")
        except ImportError:
            print("  - ⚠️ PyTorch not imported. Skipping version patch check.")
        except Exception as e:
            print(f"    ❌ Error applying PyTorch patch: {e}")
            all_success = False

        # 3. Triton installation fix
        self._fix_triton_installation()

        return all_success

    def _install_backend_requirements(self):
        """
        Install all backend dependencies from the unified requirements-backend.txt file.
        This is the single source of truth for Python packages.
        """
        requirements_file = os.path.join(self.project_root, "requirements-backend.txt")

        if not os.path.exists(requirements_file):
            print(f"❌ CRITICAL: Unified requirements file not found at {requirements_file}")
            print("   Cannot proceed with installation.")
            return False

        print(f"📦 Installing all backend dependencies from {requirements_file}...")

        # Use our robust pip detection to get the correct pip command
        pip_cmd_str = self._detect_current_pip()
        if isinstance(pip_cmd_str, str) and " -m pip" in pip_cmd_str:
            pip_cmd = pip_cmd_str.split()
        else:
            pip_cmd = [pip_cmd_str]

        print(f"🎯 Using pip command: {' '.join(pip_cmd)}")

        try:
            install_cmd = pip_cmd + ['install', '-r', requirements_file]
            subprocess.run(install_cmd, check=True, capture_output=True)
            print("✅ All backend dependencies installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print("🔥 Pip install failed. This is often due to a version conflict or a missing system library.")
            print(f"   ❌ Error installing from {requirements_file}: {e}")
            print("\n" + "="*80)
            print("🔥 STDOUT from pip:")
            print(e.stdout.decode('utf-8', errors='ignore'))
            print("\n" + "="*80)
            print("🔥 STDERR from pip:")
            print(e.stderr.decode('utf-8', errors='ignore'))
            print("="*80)
            print("\n💡 Common Fixes:")
            print("   - Check the error log above for specific package issues (e.g., 'failed building wheel').")
            print("   - Ensure you have build tools installed (e.g., `sudo apt-get install build-essential` on Debian/Ubuntu).")
            print("   - For CUDA-related errors, ensure your NVIDIA driver and CUDA toolkit versions are compatible.")
            return False

    def _fix_triton_installation(self):
        """Fix Triton installation for bitsandbytes AdamW8bit compatibility"""
        print("🔧 Installing Triton for bitsandbytes (AdamW8bit) compatibility...")

        # Use our robust pip detection for sd_scripts directory
        sd_venv_python = get_venv_python_path(self.sd_scripts_dir)
        if os.path.exists(sd_venv_python):
            # Found sd_scripts specific venv, use python -m pip for reliability
            pip_cmd = [sd_venv_python, "-m", "pip"]
            print(f"🎯 Using SD scripts venv: {sd_venv_python} -m pip")
        else:
            # Fall back to current environment pip
            current_pip = self._detect_current_pip()
            if isinstance(current_pip, str) and " -m pip" in current_pip:
                # It's a "python -m pip" command
                pip_cmd = current_pip.split()
            else:
                # It's a direct pip path
                pip_cmd = [current_pip]
            print(f"🎯 Using current environment pip: {' '.join(pip_cmd)}")

        # Test if Triton is already working in the target environment
        try:
            # Get python executable from our pip command
            if len(pip_cmd) > 2 and pip_cmd[1] == '-m':
                python_executable = pip_cmd[0]  # It's [python, -m, pip]
            else:
                python_executable = sys.executable  # Fallback

            test_cmd = [python_executable, '-c', 'import triton; print("Triton version:", triton.__version__)']
            result = subprocess.run(test_cmd, check=True, capture_output=True, text=True)
            print(f"✅ Triton already working: {result.stdout.strip()}")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ Triton not found or not working, installing...")

        # Install Triton with CUDA 12.4 compatibility (matches Kohya's PyTorch)
        triton_install_methods = [
            # Method 1: PyTorch CUDA 12.4 index (matches Kohya exactly)
            pip_cmd + ['install', '--force-reinstall', 'triton', '--index-url=https://download.pytorch.org/whl/cu124'],

            # Method 2: Force Linux x86_64 platform with CUDA 12.4
            pip_cmd + ['install', '--force-reinstall', 'triton', '--platform=linux_x86_64', '--only-binary=:all:',
             '--index-url=https://download.pytorch.org/whl/cu124'],

            # Method 3: Specific Triton version known to work with PyTorch 2.5.1
            pip_cmd + ['install', '--force-reinstall', 'triton==3.0.0'],

            # Method 4: Fallback to PyPI with no cache
            pip_cmd + ['install', '--force-reinstall', '--no-cache-dir', 'triton']
        ]

        for i, method in enumerate(triton_install_methods):
            try:
                print(f"🔄 Method {i+1}: Installing Triton...")
                subprocess.run(method, check=True, cwd=self.sd_scripts_dir if os.path.exists(self.sd_scripts_dir) else None)

                # Test if it works in the target environment
                if len(pip_cmd) > 2 and pip_cmd[1] == '-m':
                    python_executable = pip_cmd[0]
                else:
                    python_executable = sys.executable # Fallback
                test_cmd = [python_executable,
                           '-c', 'import triton; print("✅ Triton working!")']
                subprocess.run(test_cmd, check=True, capture_output=True)
                print(f"✅ Triton installation method {i+1} succeeded!")

                # Also test bitsandbytes import since that's what we really care about
                try:
                    if len(pip_cmd) > 2 and pip_cmd[1] == '-m':
                        python_executable = pip_cmd[0]
                    else:
                        python_executable = sys.executable # Fallback
                    test_bnb_cmd = [python_executable,
                                   '-c', 'import bitsandbytes; print("✅ bitsandbytes can use Triton!")']
                    subprocess.run(test_bnb_cmd, check=True, capture_output=True)
                    print("✅ bitsandbytes + Triton integration working!")
                except subprocess.CalledProcessError:
                    print("⚠️ Triton installed but bitsandbytes integration may have issues")

                return

            except subprocess.CalledProcessError as e:
                print(f"❌ Method {i+1} failed: {e}")
                continue

        print("⚠️ All Triton installation methods failed")
        print("💡 AdamW8bit will fallback to slower implementations or regular AdamW")
        print("💡 Training will still work, just without 8-bit optimization benefits")

    def _verify_installation(self):
        """Verify all components are working"""

        # Check SD scripts
        train_script = os.path.join(self.sd_scripts_dir, "train_network.py")
        if os.path.exists(train_script):
            print("   ✅ Kohya SD training scripts")
        else:
            print("   ❌ Kohya SD training scripts missing")

        # Determine the correct python path for sd_scripts venv to test imports
        if sys.platform == "win32":
            sd_venv_python = os.path.join(self.sd_scripts_dir, "venv", "Scripts", "python.exe")
        else:
            sd_venv_python = os.path.join(self.sd_scripts_dir, "venv", "bin", "python")

        # Use sd_scripts venv if it exists, otherwise use system python
        if os.path.exists(sd_venv_python):
            test_python = sd_venv_python
            print(f"   🎯 Testing imports with SD scripts venv: {test_python}")
        else:
            test_python = sys.executable
            print(f"   🎯 Testing imports with system Python: {test_python}")

        # Test imports in the target environment
        def test_import(module_name, description):
            try:
                result = subprocess.run([test_python, '-c', f'import {module_name}'],
                                      check=True, capture_output=True, text=True)
                print(f"   ✅ {description}")
                return True
            except subprocess.CalledProcessError:
                print(f"   ❌ {module_name} import failed: {description}")
                return False

        # Test LyCORIS import
        if os.path.exists(self.lycoris_dir):
            print("   ✅ LyCORIS directory available (DoRA, LoHa, LoKr, etc.)")
        else:
            print("   ❌ LyCORIS directory missing")

        # Test core package imports in the sd_scripts environment
        test_import('pytorch_optimizer', 'PyTorch Optimizer: CAME, Prodigy, etc.')
        test_import('safetensors', 'SafeTensors: model loading')

        # Test custom optimizers
        try:
            result = subprocess.run([test_python, '-c', 'import LoraEasyCustomOptimizer'],
                                  check=True, capture_output=True, text=True)
            print("   ✅ Custom optimizers: Derrian's utilities")
        except subprocess.CalledProcessError:
            print("   ❌ Custom optimizers: No module named 'LoraEasyCustomOptimizer'")

        # Test SD scripts library (most important!)
        try:
            # Add sd_scripts to Python path and test import
            import_test = f"import sys; sys.path.insert(0, '{self.sd_scripts_dir}'); import library"
            result = subprocess.run([test_python, '-c', import_test],
                                  check=True, capture_output=True, text=True)
            print("   ✅ Derrian's utilities: library module")
        except subprocess.CalledProcessError:
            print("   ❌ Derrian's utilities: No module named 'library'")

        print("   🏁 Setup verification complete!")

class ModelManager:
    def __init__(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.pretrained_model_dir = os.path.join(self.project_root, "pretrained_model")
        self.vae_dir = os.path.join(self.project_root, "vae")
        os.makedirs(self.pretrained_model_dir, exist_ok=True)
        os.makedirs(self.vae_dir, exist_ok=True)

    def _validate_url(self, url):
        if re.search(r"https:\/\/huggingface\.co\/.*(?:resolve|blob).*", url):
            return url.replace("blob", "resolve")
        elif m := re.search(r"https:\/\/civitai\.com\/models\/(\d+)", url):
            if model_version_id := re.search(r"modelVersionId=(\d+)", url):
                return f"https://civitai.com/api/download/models/{model_version_id.group(1)}"
        return url # Return original url if no match

    def download_file(self, url, dest_dir, api_token=""):
        """Download file with robust fallback chain: hf-transfer → aria2c → wget"""
        validated_url = self._validate_url(url)
        if not validated_url:
            print("Invalid URL provided.")
            return None

        filename = os.path.basename(validated_url.split('?')[0])
        destination_path = os.path.join(dest_dir, filename)

        # Check if file already exists
        if os.path.exists(destination_path) and os.path.getsize(destination_path) > 0:
            file_size = os.path.getsize(destination_path)
            print(f"✅ File already exists: {destination_path}")
            print(f"📁 Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            print(f"🕒 Last modified: {datetime.datetime.fromtimestamp(os.path.getmtime(destination_path)).strftime('%Y-%m-%d %H:%M:%S')}")
            return destination_path

        print(f"Downloading from {validated_url}...")

        # Method 1: Try aria2c first (fastest, most reliable)
        if shutil.which("aria2c"):
            print("🚀 Attempting download with aria2c...")
            try:
                header = ""
                download_url = validated_url

                parsed_url = urlparse(validated_url)
                hostname = parsed_url.hostname or ""

                # Check for civitai.com (domain or subdomains)
                if (
                    (hostname == "civitai.com" or hostname.endswith(".civitai.com"))
                    and api_token
                    and "hf" not in api_token
                ):
                    download_url = f"{validated_url}?token={api_token}"
                # Check for huggingface.co (domain or subdomains)
                elif (
                    (hostname == "huggingface.co" or hostname.endswith(".huggingface.co"))
                    and api_token
                ):
                    header = f"Authorization: Bearer {api_token}"

                command = [
                    "aria2c", download_url,
                    "--console-log-level=warn",
                    "-c", "-s", "16", "-x", "16", "-k", "10M",
                    "-d", dest_dir,
                    "-o", filename
                ]
                if header:
                    command.extend(["--header", header])

                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                for line in iter(process.stdout.readline, ''):
                    print(line, end='')

                process.stdout.close()
                return_code = process.wait()

                if return_code == 0:
                    print(f"✅ Download complete with aria2c: {destination_path}")
                    return destination_path
                else:
                    print(f"❌ aria2c failed with exit code {return_code}. Trying next method.")

            except Exception as e:
                print(f"❌ Error with aria2c: {e}. Trying next method.")
        else:
            print("⚠️ aria2c not available. Trying hf-transfer.")

        # Method 2: Try hf-transfer for Hugging Face URLs
        if "huggingface.co" in validated_url and shutil.which("hf-transfer"):
            print("🚀 Attempting download with hf-transfer...")
            try:
                env = os.environ.copy()
                if api_token:
                    env["HF_HUB_TOKEN"] = api_token

                hf_download_cmd = ["hf-transfer", "download", validated_url, "--local-dir", dest_dir]
                process = subprocess.Popen(
                    hf_download_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env
                )

                for line in iter(process.stdout.readline, ''):
                    print(line, end='')

                process.stdout.close()
                return_code = process.wait()

                if return_code == 0:
                    print(f"✅ Download complete with hf-transfer: {destination_path}")
                    return destination_path
                else:
                    print(f"❌ hf-transfer failed with exit code {return_code}. Falling back to wget.")

            except Exception as e:
                print(f"❌ Error with hf-transfer: {e}. Falling back to wget.")
        else:
            if "huggingface.co" in validated_url:
                print("⚠️ hf-transfer not available for HuggingFace URL. Falling back to wget.")

        # Method 3: Try wget as fallback
        if shutil.which("wget"):
            print("🚀 Attempting download with wget...")
            try:
                download_url = validated_url
                wget_args = ["wget", "-O", destination_path]

                if "civitai.com" in validated_url and api_token and "hf" not in api_token:
                    download_url = f"{validated_url}?token={api_token}"
                elif "huggingface.co" in validated_url and api_token:
                    wget_args.extend(["--header", f"Authorization: Bearer {api_token}"])

                wget_args.append(download_url)

                process = subprocess.Popen(
                    wget_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                for line in iter(process.stdout.readline, ''):
                    print(line, end='')

                process.stdout.close()
                return_code = process.wait()

                if return_code == 0:
                    print(f"✅ Download complete with wget: {destination_path}")
                    return destination_path
                else:
                    print(f"❌ wget failed with exit code {return_code}.")

            except Exception as e:
                print(f"❌ Error with wget: {e}")
        else:
            print("⚠️ wget not available.")

        # Method 4: Python requests as absolute fallback
        print("🚀 Attempting download with Python requests (final fallback)...")
        try:
            import requests

            headers = {}
            download_url = validated_url

            if "civitai.com" in validated_url and api_token and "hf" not in api_token:
                download_url = f"{validated_url}?token={api_token}"
            elif "huggingface.co" in validated_url and api_token:
                headers["Authorization"] = f"Bearer {api_token}"

            response = requests.get(download_url, headers=headers, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(destination_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)

            print(f"\n✅ Download complete with Python requests: {destination_path}")
            return destination_path

        except Exception as e:
            print(f"❌ Error with Python requests: {e}")

        print("💥 All download methods failed!")
        return None

    def download_model(self, model_url, model_name=None, api_token=""):
        """Download a model to the pretrained_model directory"""
        filename = model_name if model_name else os.path.basename(model_url.split('?')[0])
        if not filename.endswith(('.safetensors', '.ckpt')):
            filename += '.safetensors'

        return self.download_file(model_url, self.pretrained_model_dir, api_token)

    def download_vae(self, vae_url, vae_name=None, api_token=""):
        """Download a VAE to the vae directory"""
        filename = vae_name if vae_name else os.path.basename(vae_url.split('?')[0])
        if not filename.endswith(('.safetensors', '.ckpt')):
            filename += '.safetensors'

        return self.download_file(vae_url, self.vae_dir, api_token)

    def download_model_or_vae(self, url, download_dir):
        """
        Download a model or VAE using stored API keys from settings.

        This method automatically loads API keys from user settings and
        determines which key to use based on the URL.

        Args:
            url: URL to download from (HuggingFace or Civitai)
            download_dir: Target directory for download

        Returns:
            dict: Result dictionary with success status, file_path, and message/error
        """
        try:
            # Import here to avoid circular dependency
            import sys
            from pathlib import Path

            # Add project root to path for settings import
            project_root = Path(__file__).parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from api.routes.settings import get_api_keys

            # Load API keys from settings
            api_keys = get_api_keys()

            # Determine which API token to use based on URL
            api_token = ""
            if "huggingface.co" in url:
                api_token = api_keys.get("huggingface_token", "")
            elif "civitai.com" in url:
                api_token = api_keys.get("civitai_api_key", "")

            # Download the file
            result_path = self.download_file(url, download_dir, api_token)

            if result_path and os.path.exists(result_path):
                return {
                    "success": True,
                    "file_path": result_path,
                    "message": f"Successfully downloaded to {result_path}"
                }
            else:
                return {
                    "success": False,
                    "error": "Download failed - no file path returned"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


