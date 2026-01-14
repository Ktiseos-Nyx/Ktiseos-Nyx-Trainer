#!/usr/bin/env python3
"""
Ktiseos-Nyx-Trainer - Local Windows Dependency Installer
Installs training dependencies for local Windows users.
No GPU auto-install. User must manually install PyTorch with CUDA 12.1.

✅ NOW INCLUDES FRONTEND PROVISIONING:
   - Installs npm dependencies if node_modules/ missing
   - Builds Next.js app if .next/ missing
   - Ensures `start_services_local.bat` works reliably
"""

import argparse
import datetime
import logging
import os
import platform
import shutil
import subprocess
import sys

# Windows-specific: Ensure Colorama for colored console output
if sys.platform == "win32":
    try:
        import colorama
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "colorama"])
        import colorama

    colorama.init(autoreset=True)


class LocalWindowsInstaller:
    """Local Windows Installer for Ktiseos-Nyx-Trainer Backend + Frontend Dependencies"""

    def __init__(self, verbose=False, skip_install=False):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.verbose = verbose
        self.skip_install = skip_install

        self.setup_logging()
        self.python_cmd = sys.executable
        self.package_manager = {
            "name": "pip",
            "install_cmd": [self.python_cmd, "-m", "pip", "install"],
            "available": True,
        }

        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.derrian_dir = os.path.join(self.trainer_dir, "derrian_backend")
        self.sd_scripts_dir = os.path.join(self.derrian_dir, "sd_scripts")
        self.lycoris_dir = os.path.join(self.derrian_dir, "lycoris")

    def setup_logging(self):
        logs_dir = os.path.join(self.project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"installer_windows_local_{timestamp}.log")
        log_level = logging.DEBUG if self.verbose else logging.INFO
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger = logging.getLogger("local_windows_installer")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.log_file = log_file
        self.logger.info("Local Windows Installer logging started - Log file: %s", log_file)
        self.logger.info("Verbose mode: %s", "Enabled" if self.verbose else "Disabled")

    def print_banner(self):
        banner_lines = [
            "=" * 70,
            "Ktiseos-Nyx-Trainer - Local Windows Installer",
            "   Installing training backend (Kohya SS, LyCORIS, ONNX)",
            "=" * 70,
            f"Python: {self.python_cmd}",
            f"Package Manager: {self.package_manager['name']}",
            f"Project Root: {self.project_root}",
            f"Log File: {self.log_file}",
            "",
        ]
        for line in banner_lines:
            print(line)
            self.logger.info(line)

    def run_command(self, command, description, cwd=None, allow_failure=False):
        self.logger.info(" %s...", description)
        self.logger.debug("Command: %s", " ".join(command))
        self.logger.debug("Working directory: %s", cwd or "current")

        if not self.verbose:
            print(f" {description}...")

        # On Windows, use shell=True to support .cmd/.bat files (like npm.cmd)
        use_shell = platform.system() == "Windows"
        if use_shell:
            # Convert list to string for shell execution on Windows
            # Quote paths with spaces (like "C:\Program Files\nodejs\npm.CMD")
            quoted_command = []
            for part in command:
                part_str = str(part)
                # Quote if contains spaces and not already quoted
                if " " in part_str and not (part_str.startswith('"') and part_str.endswith('"')):
                    quoted_command.append(f'"{part_str}"')
                else:
                    quoted_command.append(part_str)
            shell_command = " ".join(quoted_command)
        else:
            shell_command = command

        try:
            if self.verbose:
                process = subprocess.Popen(
                    shell_command if use_shell else command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=cwd,
                    encoding="utf-8",
                    errors="replace",
                    shell=use_shell,
                )
                output_lines = []
                for line in iter(process.stdout.readline, ""):
                    line = line.rstrip()
                    if line:
                        print(f"   {line}")
                        self.logger.debug("OUTPUT: %s", line)
                        output_lines.append(line)
                process.stdout.close()
                return_code = process.wait()
                output = "\n".join(output_lines)
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command, output)
            else:
                result = subprocess.run(
                    shell_command if use_shell else command,
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    encoding="utf-8",
                    errors="replace",
                    shell=use_shell,
                )
                output = result.stdout

            self.logger.info(" %s successful.", description)
            self.logger.debug("Command output: %s", output)
            if not self.verbose:
                print(f" {description} successful.")
            return True

        except subprocess.CalledProcessError as e:
            error_msg = f" {description} failed."
            self.logger.error(error_msg)
            self.logger.error("Exit code: %s", e.returncode)
            self.logger.error("Command: %s", " ".join(command))
            if e.stdout:
                self.logger.error("Stdout: %s", e.stdout)
            if e.stderr:
                self.logger.error("Stderr: %s", e.stderr)
            print(error_msg)
            if self.verbose:
                print(f"   Exit code: {e.returncode}")
                if e.stderr:
                    print(f"   Error output: {e.stderr}")
            else:
                print(f"   Check log file for details: {self.log_file}")
            if not allow_failure:
                return False
            self.logger.warning("Command failed but continuing due to allow_failure=True")
            return True

        except Exception as e:
            error_msg = f" Unexpected error during {description}: {e}"
            self.logger.error(error_msg)
            print(error_msg)
            return False

    def verify_vendored_backend(self):
        print(" Verifying vendored backend...")
        self.logger.info("Checking vendored derrian_backend directory")
        if not os.path.exists(self.derrian_dir):
            error_msg = (
                f" CRITICAL: Vendored backend not found at {self.derrian_dir}\n"
                "   This repository should include trainer/derrian_backend in the clone."
            )
            self.logger.error(error_msg)
            print(error_msg)
            return False
        required_dirs = {
            "sd_scripts": self.sd_scripts_dir,
            "lycoris": self.lycoris_dir,
        }
        for name, path in required_dirs.items():
            if not os.path.exists(path):
                error_msg = f" Required directory '{name}' not found at {path}"
                self.logger.error(error_msg)
                print(error_msg)
                return False
            self.logger.info("   Found %s at %s", name, path)
            print(f"   {name} directory verified")
        print(" Vendored backend verified successfully")
        self.logger.info("Vendored backend verification complete")
        return True

    def copy_bitsandbytes_dlls_for_kohya(self):
        """Copy DLLs from bitsandbytes_windows/ into bitsandbytes/ where Kohya expects them."""
        src_dir = os.path.join(self.sd_scripts_dir, "bitsandbytes_windows")
        dest_dir = os.path.join(self.sd_scripts_dir, "bitsandbytes")

        if not os.path.exists(src_dir):
            self.logger.warning("bitsandbytes_windows source directory not found. Skipping DLL copy.")
            print("   ⚠️  Note: If using bitsandbytes, install from community Windows build:")
            print("      pip install bitsandbytes --index-url https://jllllll.github.io/bitsandbytes-windows-webui")
            return False

        os.makedirs(dest_dir, exist_ok=True)

        # Try CUDA 12.1 first (modern), fallback to CUDA 11.8 (legacy)
        dlls_to_copy = [
            "libbitsandbytes_cpu.dll",
            "libbitsandbytes_cuda121.dll",  # PRIMARY for PyTorch 2.x + CUDA 12.1
            "libbitsandbytes_cuda118.dll",  # FALLBACK for older setups
        ]

        copied = []
        for dll in dlls_to_copy:
            src = os.path.join(src_dir, dll)
            dst = os.path.join(dest_dir, dll)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied.append(dll)
                self.logger.info("Copied %s to %s", dll, dest_dir)
            else:
                self.logger.debug("DLL not found (optional): %s", src)

        if copied:
            print(f"   ✅ Copied {len(copied)} bitsandbytes DLLs for Kohya")
            self.logger.info("bitsandbytes DLLs ready for Kohya's loader")
        else:
            self.logger.warning("No bitsandbytes DLLs found in vendored folder")
            print("   ℹ️  To use 8-bit optimizers (AdamW8bit), install community Windows bitsandbytes:")
            print("      pip install bitsandbytes --index-url https://jllllll.github.io/bitsandbytes-windows-webui")

        return True

    def install_dependencies(self):
        if self.skip_install:
            print(" Skipping dependency installation (--skip-install flag)")
            self.logger.info("Skipping dependency installation due to --skip-install flag")
            return True
        requirements_file = os.path.join(self.project_root, "requirements.txt")
        if not os.path.exists(requirements_file):
            error_msg = " CRITICAL: requirements.txt not found!"
            self.logger.error(error_msg)
            print(error_msg)
            return False
        self.logger.info("Installing dependencies from: %s", requirements_file)
        install_cmd = self.package_manager["install_cmd"] + ["-r", requirements_file]
        success = self.run_command(install_cmd, f"Installing Python packages with {self.package_manager['name']}")
        if success:
            self.fix_onnx_runtime()
        return success

    def fix_onnx_runtime(self):
        self.logger.info(" Ensuring correct ONNX runtime installation for Windows...")
        print(" Ensuring correct ONNX runtime installation for Windows...")
        uninstall_cmd = [self.python_cmd, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"]
        self.run_command(uninstall_cmd, "Removing existing ONNX runtime packages")
        onnx_cmd = [self.python_cmd, "-m", "pip", "install", "onnx==1.16.1", "protobuf<4"]
        cuda12_cmd = [
            self.python_cmd,
            "-m",
            "pip",
            "install",
            "--extra-index-url",
            "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/",
            "onnxruntime-gpu==1.17.1",
        ]
        success = self.run_command(onnx_cmd, "Installing ONNX and protobuf")
        if success:
            success = self.run_command(cuda12_cmd, "Installing ONNX Runtime GPU with CUDA 12 support")
        if success:
            self.logger.info(" ONNX runtime installation completed successfully")
            print(" ONNX runtime installation completed successfully")
        else:
            self.logger.warning(" ONNX runtime installation encountered issues - will fallback to CPU")
            print(" ONNX runtime installation encountered issues - will fallback to CPU")

    def check_system_dependencies(self):
        self.logger.info(" Checking system dependencies for Windows...")
        print(" Checking system dependencies for Windows...")
        if not shutil.which("aria2c"):
            warning_msg = (
                "Windows: aria2c not found. Please install manually:\n"
                "   https://aria2.github.io/   or use 'choco install aria2' / 'scoop install aria2'"
            )
            self.logger.warning(warning_msg)
            print(f"   {warning_msg}")
        else:
            self.logger.info("aria2c: Found")
            print("   - aria2c: Found")
        return True

    def check_microsoft_store_python(self):
        """Detect and warn about Microsoft Store Python installations."""
        if "WindowsApps" in self.python_cmd and "PythonSoftwareFoundation" in self.python_cmd:
            warning_lines = [
                "\n" + "!" * 70,
                "⚠️  WARNING: Microsoft Store Python detected!",
                "!" * 70,
                "You're using Python from the Microsoft Store, which can cause issues:",
                "  - PATH conflicts with other Python installations",
                "  - Permission issues with certain packages",
                "  - Difficulty using virtual environments",
                "",
                "RECOMMENDED: Install Python from python.org instead:",
                "  1. Visit https://www.python.org/downloads/",
                "  2. Download Python 3.10, 3.11, or 3.12",
                "  3. During installation, check 'Add Python to PATH'",
                "  4. Uninstall Microsoft Store Python from Settings > Apps",
                "",
                "Current installation MAY work, but you might encounter issues.",
                "!" * 70 + "\n",
            ]
            for line in warning_lines:
                print(line)
                self.logger.warning(line)

            # Give user a chance to cancel
            print("Press Ctrl+C within 10 seconds to cancel, or wait to continue...")
            import time
            try:
                time.sleep(10)
            except KeyboardInterrupt:
                print("\nInstallation cancelled by user.")
                sys.exit(0)

    def install_pytorch(self):
        """
        Install PyTorch with CUDA 12.1 support for Windows.
        This is LOCAL INSTALLER ONLY - VastAI has PyTorch pre-installed.
        """
        try:
            import torch  # pyright: ignore

            # PyTorch is already installed - check if it has CUDA
            self.logger.info("PyTorch already installed: %s", torch.__version__)
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                self.logger.info("CUDA available: %s", cuda_version)
                print(f"\n✅ PyTorch {torch.__version__} + CUDA {cuda_version} detected")
                print("   GPU training is ready.\n")
                return True
            else:
                # CPU-only PyTorch detected - need to reinstall with CUDA
                print("\n" + "!" * 70)
                print("🚨 CPU-only PyTorch detected!")
                print("!" * 70)
                print(f"Current: PyTorch {torch.__version__} (CPU-only)")
                print("\nReinstalling PyTorch with CUDA 12.1 support...")
                print("!" * 70 + "\n")

                # Uninstall CPU version
                uninstall_cmd = [self.python_cmd, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"]
                self.run_command(uninstall_cmd, "Uninstalling CPU-only PyTorch")

        except ImportError:
            # PyTorch not installed - install with CUDA
            print("\n" + "!" * 70)
            print("⚠️  PyTorch not found - installing with CUDA 12.1 support")
            print("!" * 70 + "\n")

        # Install PyTorch with CUDA 12.1
        pytorch_cmd = [
            self.python_cmd,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ]

        success = self.run_command(pytorch_cmd, "Installing PyTorch with CUDA 12.1")

        if success:
            # Verify installation
            try:
                import torch  # pyright: ignore
                if torch.cuda.is_available():
                    print(f"\n✅ PyTorch {torch.__version__} + CUDA {torch.version.cuda} installed successfully!")
                    print("   GPU training is ready.\n")
                    self.logger.info("PyTorch with CUDA installed successfully")
                else:
                    print("\n⚠️  PyTorch installed but CUDA not available. Check your NVIDIA drivers.")
                    self.logger.warning("PyTorch installed but CUDA not available")
            except ImportError:
                print("\n❌ PyTorch installation failed.")
                self.logger.error("PyTorch installation failed")
                return False

        return success

    # =============== NEW FRONTEND METHODS ===============
    def get_npm_executable(self):
        """Find npm executable, handling Windows .cmd files and custom install locations."""
        is_win = platform.system() == "Windows"

        # 1. Try the easy way
        npm_path = shutil.which("npm")
        if npm_path:
            return npm_path

        # 2. If on Windows, check the Node neighbor (Fixes the C: vs I: jump)
        if is_win:
            node_path = shutil.which("node")
            if node_path:
                node_dir = os.path.dirname(node_path)
                npm_cmd = os.path.join(node_dir, "npm.cmd")
                if os.path.exists(npm_cmd):
                    return npm_cmd

            # 3. Check the "Gamer" custom install spots
            common_spots = [
                os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "nodejs\\npm.cmd"),
                os.path.expandvars("%APPDATA%\\npm\\npm.cmd")
            ]
            for spot in common_spots:
                if os.path.exists(spot):
                    return spot

        return "npm"  # Fallback to string and pray

    def install_frontend_deps(self):
        """Install frontend dependencies if node_modules is missing."""
        frontend_dir = os.path.join(self.project_root, "frontend")
        if not os.path.exists(frontend_dir):
            self.logger.info("Frontend directory not found. Skipping.")
            return True

        # Find npm executable (handles Windows .cmd and custom install locations)
        npm_exe = self.get_npm_executable()
        if npm_exe == "npm" and not shutil.which("npm"):
            # Fallback failed - npm truly not found
            self.logger.warning("npm not found. Is Node.js installed?")
            print(" ⚠️  npm not found! Please install Node.js 18+ from https://nodejs.org/")
            return False

        node_modules = os.path.join(frontend_dir, "node_modules")
        if not os.path.exists(node_modules):
            self.logger.info("Installing frontend dependencies...")
            print(" 📦 Installing frontend (Next.js) dependencies...")
            success = self.run_command([npm_exe, "install"], "Installing npm packages", cwd=frontend_dir)
            return success
        else:
            self.logger.info("Frontend dependencies already installed.")
            print(" ✅ Frontend dependencies already installed.")
            return True

    def build_frontend(self):
        """Build Next.js app if .next/ is missing."""
        frontend_dir = os.path.join(self.project_root, "frontend")
        if not os.path.exists(frontend_dir):
            self.logger.info("Frontend directory not found. Skipping build.")
            return True

        # Find npm executable
        npm_exe = self.get_npm_executable()

        build_dir = os.path.join(frontend_dir, ".next")
        if not os.path.exists(build_dir):
            self.logger.info("Building Next.js production frontend...")
            print(" 🏗️  Building Next.js production frontend...")
            success = self.run_command([npm_exe, "run", "build"], "Building Next.js app", cwd=frontend_dir)
            if not success:
                self.logger.warning("Frontend build failed.")
                print(" ⚠️  Frontend build failed. Backend will still work.")
            return success
        else:
            self.logger.info("Frontend already built.")
            print(" ✅ Frontend already built.")
            return True

    # ====================================================

    def apply_special_fixes_and_installs(self):
        self.logger.info(" Applying special fixes and editable installs (Windows)...")
        editable_installs = {
            "LyCORIS": self.lycoris_dir,
            "Custom Optimizers": os.path.join(self.derrian_dir, "custom_scheduler"),
            "Kohya's SD Scripts": self.sd_scripts_dir,
        }
        for name, path in editable_installs.items():
            if os.path.exists(os.path.join(path, "setup.py")):
                self.logger.info("Installing %s in editable mode from %s", name, path)
                install_cmd = self.package_manager["install_cmd"] + ["-e", "."]
                success = self.run_command(install_cmd, f"Editable install for {name}", cwd=path, allow_failure=True)
                if not success:
                    warning_msg = f"Could not install {name} in editable mode. Training might still work."
                    self.logger.warning(warning_msg)
                    print(f"   - {warning_msg}")
            else:
                self.logger.debug("No setup.py found for %s at %s, skipping", name, path)

        # Critical: Copy DLLs into Kohya's bitsandbytes/ folder
        self.copy_bitsandbytes_dlls_for_kohya()

        # Apply PyTorch patch if needed
        self.logger.info("Checking if PyTorch version patch is needed...")
        print("   - Checking if PyTorch version patch is needed...")
        try:
            import torch  # pyright: ignore

            pytorch_version = torch.__version__
            self.logger.info("Detected PyTorch version: %s", pytorch_version)
            if pytorch_version in ["2.0.0", "2.0.1"]:
                self.logger.info("Applying patch for PyTorch %s...", pytorch_version)
                print(f"   - Applying patch for PyTorch {pytorch_version}...")
                fix_script_path = os.path.join(self.derrian_dir, "fix_torch.py")
                if os.path.exists(fix_script_path):
                    self.run_command([self.python_cmd, fix_script_path], "Applying PyTorch patch")
            else:
                info_msg = f"PyTorch version is {pytorch_version}. No patch needed."
                self.logger.info(info_msg)
                print(f"   - {info_msg}")
        except ImportError:
            warning_msg = "Could not import PyTorch. Skipping version patch check."
            self.logger.warning(warning_msg)
            print(f"   - {warning_msg}")
        except Exception as e:
            error_msg = f"Error applying PyTorch patch: {e}"
            self.logger.error(error_msg)
            print(f"   - {error_msg}")
        return True

    def run_installation(self):
        self.print_banner()

        # Check for Microsoft Store Python and warn
        self.check_microsoft_store_python()

        start_time = datetime.datetime.now()
        self.logger.info("Installation started")
        try:
            if not self.verify_vendored_backend():
                error_msg = "Halting installation due to vendored backend verification failure."
                self.logger.error(error_msg)
                print(f" {error_msg}")
                return False
            if not self.check_system_dependencies():
                error_msg = "System dependency check failed."
                self.logger.error(error_msg)
                print(f" {error_msg}")
                return False

            # CRITICAL: Install PyTorch with CUDA BEFORE requirements.txt
            # This prevents pip from auto-installing CPU-only PyTorch as a dependency
            if not self.install_pytorch():
                error_msg = "PyTorch installation failed. GPU training will not work."
                self.logger.error(error_msg)
                print(f"\n⚠️  {error_msg}")
                print("You can continue, but training will not work without PyTorch + CUDA.\n")
                # Don't halt - let user decide

            if not self.install_dependencies():
                error_msg = "Halting installation due to dependency installation failure."
                self.logger.error(error_msg)
                print(f" {error_msg}")
                return False
            if not self.apply_special_fixes_and_installs():
                warning_msg = "Some special fixes or editable installs failed."
                self.logger.warning(warning_msg)
                print(f" {warning_msg}")

            # =============== NEW: FRONTEND SETUP ===============
            # Always ensure frontend is ready, even with --skip-install
            if not self.install_frontend_deps():
                self.logger.warning("Frontend dependency installation failed.")
            if not self.build_frontend():
                self.logger.warning("Frontend build failed.")
            # ===================================================

            end_time = datetime.datetime.now()
            duration = end_time - start_time
            completion_lines = [
                "\n" + "=" * 70,
                "✅ Local Windows Installation complete!",
                f"Total time: {duration}",
                f"Package manager used: {self.package_manager['name']}",
                f"Full log available at: {self.log_file}",
                "",
                "📦 Backend dependencies installed successfully for Windows!",
                "",
                "🛡️  About Virtual Environments (STRONGLY RECOMMENDED):",
                "   - This installer does NOT create a virtual environment automatically",
                "   - Packages were installed to your Python's user site-packages",
                "   - ⚠️  For future installs, use a venv to avoid package conflicts!",
                "",
                "   How to use venv next time:",
                "     python -m venv venv",
                "     venv\\Scripts\\activate",
                "     install.bat",
                "",
                "📍 Next steps:",
                "   1. ⚠️  Check warnings above - especially PyTorch CUDA support!",
                "   2. Run 'start_services_local.bat' to start the web UI",
                "   3. Access the UI at http://localhost:3000",
                "",
                "💡 Troubleshooting:",
                "   - CUDA errors? Reinstall PyTorch with CUDA 12.1 (cu121)",
                "   - Import errors? Try python.org Python instead of MS Store",
                "   - Multiple Pythons? Check which one is first in PATH",
                "   - Need 8-bit optimizers? Install community bitsandbytes-windows",
                "=" * 70,
            ]
            for line in completion_lines:
                print(line)
                if line.strip():
                    self.logger.info(line)
            return True
        except Exception as e:
            error_msg = f"Unexpected error during installation: {e}"
            self.logger.error(error_msg)
            print(f" {error_msg}")
            print(f"Check log file for details: {self.log_file}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Ktiseos-Nyx-Trainer - Local Windows Dependency Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python installer_windows_local.py
  python installer_windows_local.py --verbose
  python installer_windows_local.py --skip-install

After installation:
  - Run start_services_local.bat to start the web UI

Logs: logs/installer_windows_local_TIMESTAMP.log
        """,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")
    args = parser.parse_args()
    try:
        installer = LocalWindowsInstaller(verbose=args.verbose, skip_install=args.skip_install)
        success = installer.run_installation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f" Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
