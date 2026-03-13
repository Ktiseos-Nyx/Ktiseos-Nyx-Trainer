#!/usr/bin/env python3
"""
Ktiseos-Nyx-Trainer - Backend Dependency Installer
Installs training dependencies (Kohya SS, LyCORIS, ONNX, etc.)
Enhanced with comprehensive logging and platform-specific fixes
"""

import argparse
import datetime
import logging
import os
import platform
import shutil
import subprocess
import sys


def get_python_command():
    """Detects the best available Python command."""
    for cmd in ["python3", "python"]:
        if shutil.which(cmd):
            try:
                result = subprocess.run([cmd, "--version"], capture_output=True, text=True, check=True)
                if result.returncode == 0 and "Python 3" in result.stdout:
                    return cmd
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
    raise RuntimeError("Python 3.10+ not found. Please install Python 3.10+")


class RemoteInstaller:
    """Unified Installer for Ktiseos-Nyx-Trainer Backend Dependencies"""

    def __init__(self, verbose=False, skip_install=False, force=False):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.verbose = verbose
        self.skip_install = skip_install
        self.force = force
        self.install_marker = os.path.join(self.project_root, "install_complete.marker")

        # Setup logging
        self.setup_logging()

        # Always use current Python executable
        # for environment-agnostic execution
        # This follows CLAUDE.md requirement: NEVER hardcode paths
        self.python_cmd = sys.executable

        # Initialize package manager with uv → pip fallback
        self.package_manager = self.detect_package_manager()

        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.derrian_dir = os.path.join(self.trainer_dir, "derrian_backend")
        self.sd_scripts_dir = os.path.join(self.derrian_dir, "sd_scripts")
        self.lycoris_dir = os.path.join(self.derrian_dir, "lycoris")

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
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        # Configure logger
        self.logger = logging.getLogger("installer")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Store log file path for reference
        self.log_file = log_file

        self.logger.info("Installer logging started - Log file: %s", log_file)
        self.logger.info("Verbose mode: %s", "Enabled" if self.verbose else "Disabled")

    def detect_package_manager(self):
        """Use pip for package installation"""
        self.logger.info("Using pip for package installation")
        return {"name": "pip", "install_cmd": [self.python_cmd, "-m", "pip", "install"], "available": True}

    def get_install_command(self, *args):
        """Get package installation command with current package manager"""
        return self.package_manager["install_cmd"] + list(args)

    def ensure_pytorch_installed(self):
        """Install PyTorch with CUDA 12.1 (for remote GPU containers)"""
        try:
            import torch  # pyright: ignore[reportMissingImports]

            self.logger.info("PyTorch already installed: %s", torch.__version__)
            return
        except ImportError:
            self.logger.info("PyTorch not found. Installing with CUDA 12.1...")
            # Use standard CUDA 12.1 (most compatible)
            cmd = [
                self.python_cmd,
                "-m",
                "pip",
                "install",
                "torch==2.4.0",
                "torchvision==0.19.0",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
            ]
            self.run_command(cmd, "Installing PyTorch with CUDA 12.1")

    def print_banner(self):
        banner_lines = [
            "=" * 70,
            "Ktiseos-Nyx-Trainer - Remote Dependency Installer",
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
            self.logger.info(line.replace("Log File: ", ""))

    def run_command(self, command, description, cwd=None, allow_failure=False):
        """Enhanced command runner with comprehensive logging"""
        self.logger.info(" %s...", description)
        self.logger.debug("Command: %s", " ".join(command))
        self.logger.debug("Working directory: %s", cwd or "current")

        if not self.verbose:
            print(f" {description}...")

        try:
            # Run command with real-time output if verbose
            if self.verbose:
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=cwd
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
                # Run command silently
                result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=cwd)
                output = result.stdout

            self.logger.info(" %s successful.", description)
            self.logger.debug("Command output: %s", output)

            if not self.verbose:
                print(" %s successful." % description)

            return True

        except subprocess.CalledProcessError as e:
            error_msg = " %s failed." % description
            self.logger.error(error_msg)
            self.logger.error("Exit code: %s", e.returncode)
            self.logger.error("Command: %s", " ".join(command))

            if hasattr(e, "stdout") and e.stdout:
                self.logger.error("Stdout: %s", e.stdout)
            if hasattr(e, "stderr") and e.stderr:
                self.logger.error("Stderr: %s", e.stderr)

            print(error_msg)
            if self.verbose:
                print(f"   Exit code: {e.returncode}")
                if hasattr(e, "stderr") and e.stderr:
                    print(f"   Error output: {e.stderr}")
            else:
                print(f"   Check log file for details: {self.log_file}")

            if not allow_failure:
                return False
            self.logger.warning("Command failed but continuing due to %s", "allow_failure=True")
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            error_msg = f" Unexpected error during %s: {e}" % description
            self.logger.error(error_msg)
            print(error_msg)
            return False

    def verify_vendored_backend(self):
        """Verify vendored backend directory exists and has required components"""
        print("Verifying vendored backend...")
        self.logger.info("Checking vendored derrian_backend directory")

        # Check if derrian_backend directory exists
        if not os.path.exists(self.derrian_dir):
            error_msg = (
                f"CRITICAL: Vendored backend not found at {self.derrian_dir}\n"
                "   This repository should include trainer/derrian_backend in the clone.\n"
                "   If you cloned this repo, the vendored backend should already be present."
            )
            self.logger.error(error_msg)
            print(error_msg)
            return False

        # Verify key subdirectories exist
        required_dirs = {
            "sd_scripts": self.sd_scripts_dir,
            "lycoris": self.lycoris_dir,
        }

        for name, path in required_dirs.items():
            if not os.path.exists(path):
                error_msg = "Required directory '%s' not found at %s" % (name, path)
                self.logger.error(error_msg)
                print(error_msg)
                return False
            self.logger.info("   Found %s at %s", name, path)
            print(f"   {name} directory verified")

        print("Vendored backend verified successfully")
        self.logger.info("Vendored backend verification complete")
        return True

    def install_dependencies(self):
        """Install Python dependencies from requirements file"""
        if self.skip_install:
            print("Skipping dependency installation (--skip-install flag)")
            self.logger.info("Skipping dependency installation due to --skip-install flag")
            return True

        # Use cloud requirements file (NO PyTorch - pre-installed in base image)
        requirements_file = os.path.join(self.project_root, "requirements_cloud.txt")
        if not os.path.exists(requirements_file):
            # Fallback to old requirements.txt for backwards compatibility
            requirements_file = os.path.join(self.project_root, "requirements.txt")
            if not os.path.exists(requirements_file):
                error_msg = f"CRITICAL: No requirements file found!"
                self.logger.error(error_msg)
                print(error_msg)
                return False
            self.logger.warning("requirements_cloud.txt not found, falling back to requirements.txt")

        self.logger.info("Installing cloud dependencies from: %s", requirements_file)

        install_cmd = self.get_install_command("-r", requirements_file)
        success = self.run_command(install_cmd, f"Installing Python packages with {self.package_manager['name']}")

        if success:
            self.verify_onnx_runtime()

        return success

    def verify_onnx_runtime(self):
        """Verify ONNX runtime is installed and can see GPU providers."""
        self.logger.info("Verifying ONNX runtime installation...")
        print("Verifying ONNX runtime installation...")

        verify_cmd = [
            self.python_cmd,
            "-c",
            "import onnxruntime as ort; print(f'ONNX Runtime {ort.__version__} — Providers: {ort.get_available_providers()}')"
        ]

        success = self.run_command(verify_cmd, "Verifying ONNX runtime import")

        if success:
            self.logger.info("ONNX runtime verification successful")
            print("ONNX runtime verification successful")
        else:
            self.logger.error("ONNX runtime verification failed — WD14 tagging may fall back to CPU")
            print("ONNX runtime verification failed — WD14 tagging may fall back to CPU")
            print("   If tagging is slow, try: pip install --upgrade onnxruntime-gpu")

    def check_system_dependencies(self):
        """Check and attempt to install required system packages like aria2c"""
        self.logger.info("Checking system dependencies...")

        # Check for aria2c
        if not shutil.which("aria2c"):
            self.logger.warning("aria2c not found. Attempting to install...")
            print("   - aria2c not found. Attempting to install...")

            system = platform.system().lower()
            self.logger.info("Detected system: %s", system)

            if system == "linux":
                is_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
                sudo_prefix = "" if is_root else "sudo "

                # Try different package managers
                # Use shell=True with proper string commands
                package_managers = [
                    (["apt", "--version"], "%sapt update && %sapt install -y aria2" % (sudo_prefix, sudo_prefix)),
                    (["yum", "--version"], "%syum install -y aria2" % sudo_prefix),
                    (["dnf", "--version"], "%sdnf install -y aria2" % sudo_prefix),
                ]

                for pm_cmd, install_cmd in package_managers:
                    try:
                        subprocess.run(pm_cmd, capture_output=True, check=True)
                        pm_name = pm_cmd[0]
                        self.logger.info("Found package manager: %s", pm_name)
                        print(f"     Installing with {pm_name}...")

                        result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True, check=True)
                        if result.returncode == 0:
                            self.logger.info("Successfully installed aria2c with %s", pm_name)
                            print("     Successfully installed aria2c")
                            break
                        else:
                            self.logger.warning("Failed to install with %s: %s", pm_name, result.stderr)
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                else:
                    warning_msg = "Could not auto-install aria2c. Please install manually:"
                    self.logger.warning(warning_msg)
                    print(f"     {warning_msg}")
                    if is_root:
                        print("        Ubuntu/Debian: apt update && apt install -y aria2")
                        print("        CentOS/RHEL: yum install -y aria2")
                    else:
                        print("        Ubuntu/Debian: sudo apt update && sudo apt install -y aria2")
                        print("        CentOS/RHEL: sudo yum install -y aria2")

            elif system == "darwin":
                warning_msg = "macOS: Please install aria2c manually: brew install aria2"
                self.logger.warning(warning_msg)
                print(f"     {warning_msg}")

            else:
                warning_msg = (
                    f"System {system} not officially supported for aria2c auto-install. Please install manually."
                )
                self.logger.warning(warning_msg)
                print(f"     {warning_msg}")
        else:
            self.logger.info("aria2c: Found")
            print("   - aria2c: Found")

        return True

    def apply_special_fixes_and_installs(self):
        self.logger.info("Applying special fixes and performing editable installs...")

        # --- Editable Installs ---
        editable_installs = {
            "LyCORIS": self.lycoris_dir,
            "Custom Optimizers": os.path.join(self.derrian_dir, "custom_scheduler"),
            "Kohya's SD Scripts": self.sd_scripts_dir,
        }

        for name, path in editable_installs.items():
            if os.path.exists(os.path.join(path, "setup.py")):
                self.logger.info("Installing %s in editable mode from %s", name, path)
                install_cmd = self.get_install_command("-e", ".")
                success = self.run_command(install_cmd, f"Editable install for {name}", cwd=path, allow_failure=True)

                if not success:
                    warning_msg = "Could not install %s in editable mode. Training might still work." % name
                    self.logger.warning(warning_msg)
                    print(f"    - {warning_msg}")
            else:
                self.logger.debug("No setup.py found for %s at %s, skipping editable install", name, path)

        # --- Platform-Specific Fixes ---

        return True

    def check_already_installed(self):
        """Check if a previous installation exists. Returns True if we should proceed."""
        if not os.path.exists(self.install_marker):
            return True

        try:
            with open(self.install_marker, "r") as f:
                prev_info = f.read().strip()
        except Exception:
            prev_info = "unknown date"

        if self.force:
            self.logger.info("Previous installation found (%s), but --force flag set. Reinstalling.", prev_info)
            print(f"\n Previous installation detected ({prev_info})")
            print(" --force flag set, proceeding with reinstall...\n")
            return True

        print("\n" + "=" * 70)
        print(" Installation already completed!")
        print("=" * 70)
        print(f"\n Previous install: {prev_info}")
        print("\n If you want to reinstall, use one of these options:")
        print("   python installer.py --force          Reinstall everything")
        print("   python installer.py --skip-install   Skip deps, reapply fixes only")
        print("=" * 70 + "\n")
        return False

    def write_install_marker(self):
        """Write marker file indicating successful installation."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.install_marker, "w") as f:
                f.write(f"Remote install completed: {timestamp}\n")
                f.write(f"Python: {self.python_cmd}\n")
            self.logger.info("Install marker written: %s", self.install_marker)
        except Exception as e:
            self.logger.warning("Could not write install marker: %s", e)

    def run_installation(self):
        """Run the complete installation process"""
        self.print_banner()

        # Check if already installed (skip with --force)
        if not self.check_already_installed():
            return True

        # Install PyTorch with CUDA 12.1 (for remote GPU containers)
        self.ensure_pytorch_installed()

        start_time = datetime.datetime.now()
        self.logger.info("Installation started")

        try:
            if not self.verify_vendored_backend():
                error_msg = "Halting installation due to vendored backend verification failure."
                self.logger.error(error_msg)
                print(f"{error_msg}")
                return False

            if not self.check_system_dependencies():
                error_msg = "System dependency check failed."
                self.logger.error(error_msg)
                print(f"{error_msg}")
                return False

            if not self.install_dependencies():
                error_msg = "Halting installation due to dependency installation failure."
                self.logger.error(error_msg)
                print(f"{error_msg}")
                return False

            if not self.apply_special_fixes_and_installs():
                warning_msg = "Some special fixes or editable installs failed."
                self.logger.warning(warning_msg)
                print(f"{warning_msg}")

            end_time = datetime.datetime.now()
            duration = end_time - start_time

            completion_lines = [
                "\n" + "=" * 70,
                "Installation complete!",
                f"Total time: {duration}",
                f"Package manager used: {self.package_manager['name']}",
                f"Full log available at: {self.log_file}",
                "",
                "Backend dependencies installed successfully!",
                "   Next steps:",
                "   - Cloud (VastAI/RunPod): Services start automatically",
                "   - Local: Run ./start_services_local.sh to start the web UI",
                "=" * 70,
            ]

            for line in completion_lines:
                print(line)
                if line.strip():
                    self.logger.info(line)

            self.write_install_marker()
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            error_msg = f"Unexpected error during installation: {e}"
            self.logger.error(error_msg)
            print(f"{error_msg}")
            print(f"Check log file for details: {self.log_file}")
            return False


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Ktiseos-Nyx-Trainer - Backend Dependency Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python installer.py                    # Normal installation
  python installer.py --verbose         # Verbose installation with detailed output
  python installer.py -v                # Short form of verbose
  python installer.py --skip-install    # Quick restart (skip dependency installation)

The installer will:
  1. Verify vendored derrian_backend directory (Kohya SS + LyCORIS)
  2. Install system dependencies (aria2c)
  3. Install Python packages for training backend
  4. Set up editable installs for development packages

After installation:
  - Cloud (VastAI/RunPod): Services start automatically
  - Local: Run ./start_services_local.sh to start the web interface

Logs are automatically saved to logs/installer_TIMESTAMP.log for debugging.
        """,
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output with detailed logging")

    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation (for quick restarts when deps already installed)",
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force reinstall even if already installed",
    )

    args = parser.parse_args()

    try:
        installer = RemoteInstaller(verbose=args.verbose, skip_install=args.skip_install, force=args.force)
        success = installer.run_installation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nInstallation interrupted by user")
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
