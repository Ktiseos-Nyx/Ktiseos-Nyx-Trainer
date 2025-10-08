#!/usr/bin/env python3
"""
LoRA Easy Training - Unified Command-Line Installer
Enhanced with uv fallback and comprehensive logging
"""

import os
import sys
import subprocess
import shutil
import platform
import logging
import argparse
import datetime
from pathlib import Path

def get_python_command():
    """Detects the best available Python command."""
    for cmd in ['python3', 'python']:
        if shutil.which(cmd):
            try:
                result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
                if result.returncode == 0 and 'Python 3' in result.stdout:
                    return cmd
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
    raise RuntimeError("❌ Python 3.10+ not found. Please install Python 3.10+")

class UnifiedInstaller:
    def __init__(self, verbose=False):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.verbose = verbose
        
        # Setup logging
        self.setup_logging()
        
        # Always use current Python executable for environment-agnostic execution
        # This follows CLAUDE.md requirement: NEVER hardcode paths or environment assumptions
        self.python_cmd = sys.executable
        
        # Initialize package manager with uv → pip fallback
        self.package_manager = self.detect_package_manager()
        
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.derrian_dir = os.path.join(self.trainer_dir, "derrian_backend")
        self.sd_scripts_dir = os.path.join(self.derrian_dir, "sd_scripts")
        self.lycoris_dir = os.path.join(self.derrian_dir, "lycoris")
        
        self.submodules = {
            'derrian_backend': {
                'url': 'https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend.git',
                'path': self.derrian_dir,
                'description': "Derrian's backend with Kohya scripts, LyCORIS, and custom optimizers"
            }
        }

    def setup_logging(self):
        """Setup comprehensive logging system"""
        # Create logs directory
        logs_dir = os.path.join(self.project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate timestamp for log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"installer_{timestamp}.log")
        
        # Configure logging
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Configure logger
        self.logger = logging.getLogger('installer')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Store log file path for reference
        self.log_file = log_file
        
        self.logger.info(f"Installer logging started - Log file: {log_file}")
        self.logger.info(f"Verbose mode: {'Enabled' if self.verbose else 'Disabled'}")

    def detect_package_manager(self):
        """Use pip for package installation"""
        self.logger.info("📦 Using pip for package installation")
        return {
            'name': 'pip',
            'install_cmd': [self.python_cmd, '-m', 'pip', 'install'],
            'available': True
        }

    def get_install_command(self, *args):
        """Get package installation command with current package manager"""
        return self.package_manager['install_cmd'] + list(args)

    def print_banner(self):
        banner_lines = [
            "=" * 70,
            "🚀 LoRA Easy Training - Unified Command-Line Installer",
            "   Enhanced with comprehensive logging and error handling",
            "=" * 70,
            f"🐍 Using Python: {self.python_cmd}",
            f"📦 Package Manager: {self.package_manager['name']}",
            f"🔧 Project Root: {self.project_root}",
            f"📝 Log File: {self.log_file}",
            ""
        ]
        
        for line in banner_lines:
            print(line)
            self.logger.info(line.replace("📝 Log File: ", ""))

    def run_command(self, command, description, cwd=None, allow_failure=False):
        """Enhanced command runner with comprehensive logging"""
        self.logger.info(f"🔄 {description}...")
        self.logger.debug(f"Command: {' '.join(command)}")
        self.logger.debug(f"Working directory: {cwd or 'current'}")
        
        if not self.verbose:
            print(f"🔄 {description}...")
        
        try:
            # Run command with real-time output if verbose
            if self.verbose:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=cwd
                )
                
                output_lines = []
                for line in iter(process.stdout.readline, ''):
                    line = line.rstrip()
                    if line:
                        print(f"   {line}")
                        self.logger.debug(f"OUTPUT: {line}")
                        output_lines.append(line)
                
                process.stdout.close()
                return_code = process.wait()
                output = '\n'.join(output_lines)
                
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command, output)
            else:
                # Run command silently
                result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=cwd)
                output = result.stdout
                
            self.logger.info(f"✅ {description} successful.")
            self.logger.debug(f"Command output: {output}")
            
            if not self.verbose:
                print(f"✅ {description} successful.")
            
            return True
            
        except subprocess.CalledProcessError as e:
            error_msg = f"❌ {description} failed."
            self.logger.error(error_msg)
            self.logger.error(f"Exit code: {e.returncode}")
            self.logger.error(f"Command: {' '.join(command)}")
            
            if hasattr(e, 'stdout') and e.stdout:
                self.logger.error(f"Stdout: {e.stdout}")
            if hasattr(e, 'stderr') and e.stderr:
                self.logger.error(f"Stderr: {e.stderr}")
                
            print(error_msg)
            if self.verbose:
                print(f"   Exit code: {e.returncode}")
                if hasattr(e, 'stderr') and e.stderr:
                    print(f"   Error output: {e.stderr}")
            else:
                print(f"   Check log file for details: {self.log_file}")
            
            if not allow_failure:
                return False
            else:
                self.logger.warning(f"Command failed but continuing due to allow_failure=True")
                return True
                
        except Exception as e:
            error_msg = f"❌ Unexpected error during {description}: {e}"
            self.logger.error(error_msg)
            print(error_msg)
            return False

    def setup_submodules(self):
        print("🔗 Cloning and initializing all submodules...")
        os.makedirs(self.trainer_dir, exist_ok=True)
        
        for name, config in self.submodules.items():
            path = config['path']
            url = config['url']
            
            if os.path.exists(path) and os.listdir(path):
                print(f"   - Submodule '{name}' already exists. Skipping clone.")
                continue

            if not self.run_command(['git', 'clone', url, path], f"Cloning {name}"):
                return False

        # Initialize nested submodules (sd_scripts, lycoris)
        if not self.run_command(['git', 'submodule', 'update', '--init', '--recursive'], "Initializing nested submodules", cwd=self.derrian_dir):
            return False
            
        return True

    def install_dependencies(self):
        """Install dependencies with uv → pip fallback"""
        requirements_file = os.path.join(self.project_root, "requirements-backend.txt")
        if not os.path.exists(requirements_file):
            error_msg = f"❌ CRITICAL: requirements-backend.txt not found!"
            self.logger.error(error_msg)
            print(error_msg)
            return False
        
        self.logger.info(f"Installing dependencies from: {requirements_file}")
        
        # Try with current package manager first
        install_cmd = self.get_install_command('-r', requirements_file)
        success = self.run_command(install_cmd, f"Installing Python packages with {self.package_manager['name']}")
        
        # If uv failed, fallback to pip
        if not success and self.package_manager['name'] == 'uv':
            self.logger.warning("uv installation failed, falling back to pip")
            print("⚠️ uv installation failed, falling back to pip...")
            
            # Update package manager to pip
            self.package_manager = {
                'name': 'pip',
                'install_cmd': [self.python_cmd, '-m', 'pip', 'install'],
                'available': True
            }
            
            # Retry with pip
            install_cmd = self.get_install_command('-r', requirements_file)
            success = self.run_command(install_cmd, "Installing Python packages with pip (fallback)")
        
        # Post-installation: Force correct CUDA 12 ONNX runtime
        if success:
            self.fix_onnx_runtime()
        
        return success

    def fix_onnx_runtime(self):
        """Force install correct CUDA 12 ONNX runtime to prevent version conflicts"""
        self.logger.info("🔧 Ensuring correct CUDA 12 ONNX runtime installation...")
        print("🔧 Ensuring correct CUDA 12 ONNX runtime installation...")
        
        # Uninstall any existing onnxruntime packages to prevent conflicts
        uninstall_cmd = [self.python_cmd, '-m', 'pip', 'uninstall', '-y', 'onnxruntime', 'onnxruntime-gpu']
        self.run_command(uninstall_cmd, "Removing existing ONNX runtime packages")
        
        # Install correct CUDA 12 version
        onnx_cmd = [
            self.python_cmd, '-m', 'pip', 'install', 
            'onnx==1.16.1',
            'protobuf<4'
        ]
        
        cuda12_cmd = [
            self.python_cmd, '-m', 'pip', 'install',
            '--extra-index-url', 'https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/',
            'onnxruntime-gpu==1.17.1'
        ]
        
        success = self.run_command(onnx_cmd, "Installing ONNX and protobuf")
        if success:
            success = self.run_command(cuda12_cmd, "Installing ONNX Runtime GPU with CUDA 12 support")
            
        if success:
            self.logger.info("✅ ONNX runtime CUDA 12 installation completed successfully")
            print("✅ ONNX runtime CUDA 12 installation completed successfully")
        else:
            self.logger.warning("⚠️ ONNX runtime installation encountered issues - will fallback to CPU")
            print("⚠️ ONNX runtime installation encountered issues - will fallback to CPU")

    def check_system_dependencies(self):
        """Check and attempt to install required system packages like aria2c"""
        self.logger.info("🔧 Checking system dependencies...")
        
        # Check for aria2c
        if not shutil.which('aria2c'):
            self.logger.warning("aria2c not found. Attempting to install...")
            print("   - aria2c not found. Attempting to install...")
            
            system = platform.system().lower()
            self.logger.info(f"Detected system: {system}")
            
            if system == "linux":
                # Try different package managers
                package_managers = [
                    (['apt', '--version'], ['sudo', 'apt', 'update', '&&', 'sudo', 'apt', 'install', '-y', 'aria2']),
                    (['yum', '--version'], ['sudo', 'yum', 'install', '-y', 'aria2']),
                    (['dnf', '--version'], ['sudo', 'dnf', 'install', '-y', 'aria2'])
                ]
                
                for pm_cmd, install_cmd in package_managers:
                    try:
                        subprocess.run(pm_cmd, capture_output=True, check=True)
                        pm_name = pm_cmd[0]
                        self.logger.info(f"Found package manager: {pm_name}")
                        print(f"     ✅ Installing with {pm_name}...")
                        
                        result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True)
                        if result.returncode == 0:
                            self.logger.info(f"Successfully installed aria2c with {pm_name}")
                            break
                        else:
                            self.logger.warning(f"Failed to install with {pm_name}: {result.stderr}")
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                else:
                    warning_msg = "Could not auto-install aria2c. Please install manually:"
                    self.logger.warning(warning_msg)
                    print(f"     ⚠️  {warning_msg}")
                    print("        Ubuntu/Debian: sudo apt install aria2")
                    print("        CentOS/RHEL: sudo yum install aria2")
            
            elif system == "darwin":
                if shutil.which('brew'):
                    self.logger.info("Installing aria2c with Homebrew...")
                    print("     ✅ Installing with Homebrew...")
                    result = subprocess.run(['brew', 'install', 'aria2'], capture_output=True, text=True)
                    if result.returncode == 0:
                        self.logger.info("Successfully installed aria2c with Homebrew")
                    else:
                        self.logger.warning(f"Homebrew installation failed: {result.stderr}")
                else:
                    warning_msg = "Please install aria2c: brew install aria2"
                    self.logger.warning(warning_msg)
                    print(f"     ⚠️  {warning_msg}")
            
            elif system == "windows":
                warning_msg = "Windows: Please install aria2c manually or use package manager"
                self.logger.warning(warning_msg)
                print(f"     ⚠️  {warning_msg}")
        else:
            self.logger.info("aria2c: Found")
            print("   - aria2c: ✅ Found")
        
        return True

    def apply_special_fixes_and_installs(self):
        self.logger.info("🔧 Applying special fixes and performing editable installs...")
        
        # --- Editable Installs ---
        editable_installs = {
            "LyCORIS": self.lycoris_dir,
            "Custom Optimizers": os.path.join(self.derrian_dir, "custom_scheduler"),
            "Kohya's SD Scripts": self.sd_scripts_dir
        }
        
        for name, path in editable_installs.items():
            if os.path.exists(os.path.join(path, 'setup.py')):
                self.logger.info(f"Installing {name} in editable mode from {path}")
                install_cmd = self.get_install_command('-e', '.')
                success = self.run_command(install_cmd, f"Editable install for {name}", cwd=path, allow_failure=True)
                
                if not success:
                    warning_msg = f"Could not install {name} in editable mode. Training might still work."
                    self.logger.warning(warning_msg)
                    print(f"   - ⚠️  {warning_msg}")
            else:
                self.logger.debug(f"No setup.py found for {name} at {path}, skipping editable install")
        
        # --- Platform-Specific Fixes ---
        if platform.system() == "Windows":
            self.logger.info("Applying Windows-specific fix for bitsandbytes...")
            print("   - Applying Windows-specific fix for bitsandbytes...")
            try:
                bnb_src_dir = os.path.join(self.sd_scripts_dir, 'bitsandbytes_windows')
                result = subprocess.run([self.python_cmd, '-c', 'import site; print(site.getsitepackages()[0])'], capture_output=True, text=True, check=True)
                site_packages = result.stdout.strip()
                bnb_dest_dir = os.path.join(site_packages, 'bitsandbytes')
                
                self.logger.debug(f"Bitsandbytes source: {bnb_src_dir}")
                self.logger.debug(f"Bitsandbytes destination: {bnb_dest_dir}")
                
                if os.path.exists(bnb_dest_dir):
                    # This can be improved with logic to select the correct CUDA version DLL
                    dll_to_copy = 'libbitsandbytes_cuda118.dll' 
                    src_file = os.path.join(bnb_src_dir, dll_to_copy)
                    dest_file = os.path.join(bnb_dest_dir, 'libbitsandbytes_cudaall.dll')
                    if os.path.exists(src_file):
                        shutil.copy2(src_file, dest_file)
                        success_msg = f"Copied {dll_to_copy} to {dest_file}"
                        self.logger.info(success_msg)
                        print(f"     ✅ {success_msg}")
                    else:
                        error_msg = f"Could not find source DLL: {src_file}"
                        self.logger.warning(error_msg)
                        print(f"     ⚠️ {error_msg}")
                else:
                    error_msg = "bitsandbytes directory not found in site-packages. Cannot apply fix."
                    self.logger.warning(error_msg)
                    print(f"     ⚠️ {error_msg}")
            except Exception as e:
                error_msg = f"Error applying bitsandbytes fix: {e}"
                self.logger.error(error_msg)
                print(f"     ❌ {error_msg}")

        # --- PyTorch version file fix ---
        self.logger.info("Checking if PyTorch version patch is needed...")
        print("   - Checking if PyTorch version patch is needed...")
        try:
            import torch
            pytorch_version = torch.__version__
            self.logger.info(f"Detected PyTorch version: {pytorch_version}")
            
            if pytorch_version in ["2.0.0", "2.0.1"]:
                self.logger.info(f"Applying patch for PyTorch {pytorch_version}...")
                print(f"   - Applying patch for PyTorch {pytorch_version}...")
                fix_script_path = os.path.join(self.derrian_dir, 'fix_torch.py')
                if os.path.exists(fix_script_path):
                    self.run_command([self.python_cmd, fix_script_path], "Applying PyTorch patch")
            else:
                info_msg = f"PyTorch version is {pytorch_version}. No patch needed."
                self.logger.info(info_msg)
                print(f"   - {info_msg}")
        except ImportError:
            warning_msg = "Could not import PyTorch. Skipping version patch check."
            self.logger.warning(warning_msg)
            print(f"   - ⚠️  {warning_msg}")
        except Exception as e:
            error_msg = f"Error applying PyTorch patch: {e}"
            self.logger.error(error_msg)
            print(f"   - ❌ {error_msg}")
        
        return True

    def fix_cuda_symlinks(self):
        """Auto-fix ONNX CUDA library symlink issues"""
        self.logger.info("Checking for ONNX CUDA library symlink issues...")
        print("🔗 Checking for ONNX CUDA library symlink issues...")
        
        try:
            # Check multiple possible CUDA library locations
            possible_cuda_dirs = [
                "/usr/local/cuda/lib64",
                "/usr/local/cuda-12/lib64", 
                "/usr/local/cuda-11/lib64",
                "/opt/cuda/lib64",
                "/usr/lib/x86_64-linux-gnu"  # Ubuntu/Debian system location
            ]
            
            cuda_lib_dir = None
            for dir_path in possible_cuda_dirs:
                if os.path.exists(dir_path):
                    cuda_lib_dir = dir_path
                    break
                    
            if not cuda_lib_dir:
                self.logger.info("CUDA library directory not found. Skipping symlink fix.")
                print("   - No CUDA installation detected. Skipping.")
                return True
                
            print(f"   📁 Using CUDA library directory: {cuda_lib_dir}")
                
            # Find available CUDA libraries - check for both libcublas and libcublasLt
            import glob
            
            # Check for all CUDA library types that ONNX needs
            cuda_libraries = {
                'libcublas': glob.glob(f"{cuda_lib_dir}/libcublas.so.*"),
                'libcublasLt': glob.glob(f"{cuda_lib_dir}/libcublasLt.so.*"), 
                'libcufft': glob.glob(f"{cuda_lib_dir}/libcufft.so.*"),
                'libcurand': glob.glob(f"{cuda_lib_dir}/libcurand.so.*"),
                'libcusparse': glob.glob(f"{cuda_lib_dir}/libcusparse.so.*"),
                'libcusolver': glob.glob(f"{cuda_lib_dir}/libcusolver.so.*"),
                # Add critical CUDA runtime library that ONNX specifically needs
                'libcudart': glob.glob(f"{cuda_lib_dir}/libcudart.so.*"),
                # Add cuDNN libraries if available
                'libcudnn': glob.glob(f"{cuda_lib_dir}/libcudnn.so.*")
            }
            
            # Check if any libraries were found
            found_libraries = {name: libs for name, libs in cuda_libraries.items() if libs}
            if not found_libraries:
                self.logger.info("No CUDA libraries found for ONNX symlink fix. Skipping.")
                print("   - No CUDA libraries found. Skipping.")
                return True
            
            created_links = []
            
            # ONNX commonly needed version targets - include specific versions ONNX looks for
            common_versions = ['10', '11', '12', '11.0', '11.2', '11.8', '12.0', '12.1', '12.2']
            
            # Process each library type dynamically
            for lib_name, lib_files in found_libraries.items():
                lib_files.sort(reverse=True)  # Get latest version
                latest_lib = lib_files[0]
                version = latest_lib.split('.so.')[-1] if '.so.' in latest_lib else 'unknown'
                print(f"   - Found {lib_name} version {version}")
                
                # Generate symlink targets for this library
                targets = []
                for ver in common_versions:
                    targets.extend([
                        f"{cuda_lib_dir}/{lib_name}.so.{ver}",
                        f"/usr/lib/x86_64-linux-gnu/{lib_name}.so.{ver}"
                    ])
                
                created_links.extend(self._create_cuda_symlinks(latest_lib, targets))
            
            if created_links:
                self.logger.info(f"Created {len(created_links)} CUDA symlinks for ONNX compatibility")
                print(f"   🎉 Created {len(created_links)} CUDA symlinks for ONNX compatibility")
                return True
            else:
                print(f"   - All symlinks already exist or no symlinks needed")
                return True
                
        except Exception as e:
            error_msg = f"Error fixing CUDA symlinks: {e}"
            self.logger.error(error_msg)
            print(f"   ❌ {error_msg}")
            return False

    def _create_cuda_symlinks(self, source_lib, target_list):
        """Helper function to create CUDA library symlinks"""
        created_links = []
        
        for target in target_list:
            try:
                # Skip if symlink already exists and points correctly
                if os.path.islink(target):
                    if os.readlink(target) == source_lib:
                        print(f"   ✅ Symlink already exists: {target} -> {source_lib}")
                        continue
                    else:
                        # Remove bad symlink
                        os.unlink(target)
                        
                # Skip if regular file exists (don't overwrite)
                if os.path.exists(target) and not os.path.islink(target):
                    print(f"   - Regular file exists, skipping: {target}")
                    continue
                    
                # Create directory if needed (for /usr/lib paths)
                target_dir = os.path.dirname(target)
                if not os.path.exists(target_dir):
                    print(f"   - Directory {target_dir} doesn't exist, skipping symlink")
                    continue
                    
                # Create symlink
                os.symlink(source_lib, target)
                created_links.append(target)
                print(f"   ✅ Created symlink: {target} -> {source_lib}")
                
            except PermissionError:
                print(f"   ⚠️ Permission denied creating symlink: {target}")
                continue
            except Exception as e:
                print(f"   ⚠️ Failed to create symlink {target}: {e}")
                continue
        
        return created_links

    def run_installation(self):
        """Run the complete installation process"""
        self.print_banner()
        
        start_time = datetime.datetime.now()
        self.logger.info("Installation started")
        
        try:
            if not self.setup_submodules():
                error_msg = "Halting installation due to submodule setup failure."
                self.logger.error(error_msg)
                print(f"❌ {error_msg}")
                return False

            if not self.check_system_dependencies():
                error_msg = "System dependency check failed."
                self.logger.error(error_msg)
                print(f"❌ {error_msg}")
                return False

            if not self.install_dependencies():
                error_msg = "Halting installation due to dependency installation failure."
                self.logger.error(error_msg)
                print(f"❌ {error_msg}")
                return False
                
            if not self.apply_special_fixes_and_installs():
                warning_msg = "Some special fixes or editable installs failed."
                self.logger.warning(warning_msg)
                print(f"⚠️ {warning_msg}")

            # Auto-fix ONNX CUDA symlink issues
            if not self.fix_cuda_symlinks():
                warning_msg = "CUDA symlink fixes failed (non-critical)."
                self.logger.warning(warning_msg)
                print(f"⚠️ {warning_msg}")

            end_time = datetime.datetime.now()
            duration = end_time - start_time
            
            completion_lines = [
                "\n" + "=" * 70,
                "✅ Installation complete!",
                f"⏱️ Total time: {duration}",
                f"📦 Package manager used: {self.package_manager['name']}",
                f"📝 Full log available at: {self.log_file}",
                "",
                "🚀 You can now start Jupyter and use the training notebooks.",
                "   Run: jupyter notebook",
                "=" * 70
            ]
            
            for line in completion_lines:
                print(line)
                if line.strip():
                    self.logger.info(line)
                    
            return True
            
        except Exception as e:
            error_msg = f"Unexpected error during installation: {e}"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")
            print(f"📝 Check log file for details: {self.log_file}")
            return False

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="LoRA Easy Training - Unified Command-Line Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python installer.py                    # Normal installation
  python installer.py --verbose         # Verbose installation with detailed output
  python installer.py -v                # Short form of verbose

The installer will:
  1. Clone/update the derrian_backend submodule
  2. Install system dependencies (aria2c)
  3. Install Python packages using uv (if available) or pip
  4. Apply platform-specific fixes
  5. Set up editable installs for development packages

Logs are automatically saved to logs/installer_TIMESTAMP.log for debugging.
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with detailed logging'
    )
    
    args = parser.parse_args()
    
    try:
        installer = UnifiedInstaller(verbose=args.verbose)
        success = installer.run_installation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()