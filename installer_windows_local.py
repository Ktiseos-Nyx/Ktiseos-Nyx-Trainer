#!/usr/bin/env python3
"""
Ktiseos-Nyx-Trainer - Local Windows Dependency Installer
Installs training dependencies for local Windows users.
No GPU auto-install. User must manually install PyTorch with CUDA 11.8.

✅ NOW INCLUDES FRONTEND PROVISIONING:
   - Installs npm dependencies if node_modules/ missing
   - Builds Next.js app if .next/ missing
   - Ensures `start_services_local.bat` works reliably
"""

import argparse
import datetime
import logging
import os
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

        try:
            if self.verbose:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=cwd,
                    encoding="utf-8",
                    errors="replace",
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
                    command, check=True, capture_output=True, text=True, cwd=cwd, encoding="utf-8", errors="replace"
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
            return False

        os.makedirs(dest_dir, exist_ok=True)

        # Kohya's hardcoded (or patched) loader expects cuda118.dll
        dlls_to_copy = [
            "libbitsandbytes_cpu.dll",
            "libbitsandbytes_cuda118.dll",  # PRIMARY for PyTorch 2.x on Windows
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
                self.logger.warning("DLL not found: %s", src)

        if copied:
            print(f"   ✅ Copied {len(copied)} bitsandbytes DLLs for Kohya")
            self.logger.info("bitsandbytes DLLs ready for Kohya's loader")
        else:
            self.logger.warning("No bitsandbytes DLLs copied — GPU training will fail")

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

    def ensure_pytorch_installed(self):
        """Do NOT install PyTorch. Just verify or warn."""
        try:
            import torch  # pyright: ignore

            self.logger.info(" PyTorch already installed: %s", torch.__version__)
            if torch.cuda.is_available():
                self.logger.info(" CUDA available: %s", torch.version.cuda)
                print(f" PyTorch {torch.__version__} + CUDA {torch.version.cuda} detected")
            else:
                print(f" PyTorch {torch.__version__} (CPU-only) detected")
        except ImportError:
            self.logger.warning(" PyTorch not found.")
            print(" PyTorch not found.")
            print("   → Please install PyTorch with CUDA 11.8 from https://pytorch.org/get-started/locally/  ")
            print("   → Select 'Windows', 'pip', 'CUDA 11.8'")
        return True

    # =============== NEW FRONTEND METHODS ===============
    def install_frontend_deps(self):
        """Install frontend dependencies if node_modules is missing."""
        frontend_dir = os.path.join(self.project_root, "frontend")
        if not os.path.exists(frontend_dir):
            self.logger.info("Frontend directory not found. Skipping.")
            return True

        if not shutil.which("npm"):
            self.logger.warning("npm not found. Is Node.js installed?")
            print(" ⚠️  npm not found! Please install Node.js 18+ from https://nodejs.org/")
            return False

        node_modules = os.path.join(frontend_dir, "node_modules")
        if not os.path.exists(node_modules):
            self.logger.info("Installing frontend dependencies...")
            print(" 📦 Installing frontend (Next.js) dependencies...")
            success = self.run_command(["npm", "install"], "Installing npm packages", cwd=frontend_dir)
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

        build_dir = os.path.join(frontend_dir, ".next")
        if not os.path.exists(build_dir):
            self.logger.info("Building Next.js production frontend...")
            print(" 🏗️  Building Next.js production frontend...")
            success = self.run_command(["npm", "run", "build"], "Building Next.js app", cwd=frontend_dir)
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
        self.ensure_pytorch_installed()
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
                "Local Windows Installation complete!",
                f"Total time: {duration}",
                f"Package manager used: {self.package_manager['name']}",
                f"Full log available at: {self.log_file}",
                "",
                "Backend dependencies installed successfully for Windows!",
                "   Next steps:",
                "   - Local: Run start_services_local.bat to start the web UI",
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
