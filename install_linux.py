#!/usr/bin/env python3
"""
Ktiseos-Nyx-Trainer - Local Linux Installer
For local development on Ubuntu/Debian/Fedora with NVIDIA GPU.
Installs CUDA 12.1 dependencies. Does NOT assume root or cloud environment.

✅ NOW INCLUDES FRONTEND PROVISIONING:
   - Installs npm dependencies if node_modules/ missing
   - Builds Next.js app if .next/ missing
   - Ensures ./start_services_local.sh works reliably
"""

import argparse
import datetime
import logging
import os
import shutil
import subprocess
import sys


class LocalLinuxInstaller:
    def __init__(self, verbose=False, skip_install=False, force=False):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.verbose = verbose
        self.skip_install = skip_install
        self.force = force
        self.install_marker = os.path.join(self.project_root, ".install_complete")
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
        log_file = os.path.join(logs_dir, f"installer_local_linux_{timestamp}.log")
        log_level = logging.DEBUG if self.verbose else logging.INFO
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger = logging.getLogger("local_linux_installer")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.log_file = log_file
        self.logger.info("Local Linux Installer started - Log: %s", log_file)

    def print_banner(self):
        print("=" * 70)
        print("Ktiseos-Nyx-Trainer - Local Linux Installer")
        print("  For NVIDIA GPU development (CUDA 12.1)")
        print("=" * 70)

    def run_command(self, command, description, cwd=None, allow_failure=False):
        self.logger.info(" %s...", description)
        if not self.verbose:
            print(f" {description}...")
        try:
            if self.verbose:
                subprocess.run(command, check=True, cwd=cwd, text=True)
            else:
                subprocess.run(command, check=True, cwd=cwd, capture_output=True, text=True)
            self.logger.info(" %s successful.", description)
            if not self.verbose:
                print(f" ✅ {description} successful.")
            return True
        except subprocess.CalledProcessError as e:
            print(f" ❌ {description} failed.")
            if not self.verbose:
                print(f"    Check log: {self.log_file}")
            self.logger.error("%s failed: %s", description, e)
            return False if not allow_failure else True
        except Exception as e:
            print(f" ❌ Unexpected error during {description}: {e}")
            self.logger.error("Unexpected error: %s", e)
            return False

    def has_nvidia_gpu(self):
        """Check if NVIDIA GPU is available (via nvidia-smi or CUDA)"""
        if shutil.which("nvidia-smi"):
            return True
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def verify_vendored_backend(self):
        if not os.path.exists(self.derrian_dir):
            print(f" ❌ Vendored backend missing: {self.derrian_dir}")
            return False
        for name, path in [("sd_scripts", self.sd_scripts_dir), ("lycoris", self.lycoris_dir)]:
            if not os.path.exists(path):
                print(f" ❌ Required dir '{name}' missing: {path}")
                return False
        print(" ✅ Vendored backend verified")
        return True

    def install_dependencies(self):
        if self.skip_install:
            print(" ⏩ Skipping dependency installation")
            return True
        # Use Linux-specific requirements file (includes PyTorch with CUDA 12.1)
        req_file = os.path.join(self.project_root, "requirements_linux.txt")
        if not os.path.exists(req_file):
            print(f" ❌ {req_file} not found")
            return False

        # Add CUDA 12.1 index for PyTorch installation
        cmd = [
            self.python_cmd, "-m", "pip", "install",
            "-r", req_file,
            "--extra-index-url", "https://download.pytorch.org/whl/cu121"
        ]
        return self.run_command(cmd, "Installing Python dependencies")

    def install_pytorch_cuda121(self):
        if self.has_nvidia_gpu():
            print(" 📦 Installing PyTorch with CUDA 12.1...")
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
            return self.run_command(cmd, "Installing PyTorch CUDA 12.1")
        else:
            print(" ⚠️ No NVIDIA GPU detected — skipping CUDA PyTorch")
            print("    Install manually if needed: pip install torch torchvision")
            return True

    def check_aria2c(self):
        if not shutil.which("aria2c"):
            print(" ⚠️ aria2c not found")
            print("    Required for dataset downloads. Install with:")
            print("      Ubuntu/Debian: sudo apt install aria2")
            print("      Fedora: sudo dnf install aria2")
        else:
            print(" ✅ aria2c found")

    def apply_editable_installs(self):
        for name, path in [
            ("LyCORIS", self.lycoris_dir),
            ("Custom Optimizers", os.path.join(self.derrian_dir, "custom_scheduler")),
            ("Kohya SD Scripts", self.sd_scripts_dir),
        ]:
            if os.path.exists(os.path.join(path, "setup.py")):
                cmd = [self.python_cmd, "-m", "pip", "install", "-e", "."]
                self.run_command(cmd, f"Editable install: {name}", cwd=path, allow_failure=True)

    # =============== NEW FRONTEND METHODS ===============
    def install_frontend_deps(self):
        """Install frontend dependencies if node_modules is missing."""
        frontend_dir = os.path.join(self.project_root, "frontend")
        if not os.path.exists(frontend_dir):
            self.logger.info("Frontend directory not found. Skipping.")
            return True

        if not shutil.which("npm"):
            print(" ⚠️ npm not found. Please install Node.js 18+ (e.g., via nvm or system package).")
            self.logger.warning("npm not found. Frontend setup skipped.")
            return False

        node_modules = os.path.join(frontend_dir, "node_modules")
        if not os.path.exists(node_modules):
            print(" 📦 Installing frontend (Next.js) dependencies...")
            success = self.run_command(["npm", "install"], "Installing frontend dependencies", cwd=frontend_dir)
            return success
        else:
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
            print(" 🏗️ Building Next.js production frontend...")
            success = self.run_command(["npm", "run", "build"], "Building Next.js app", cwd=frontend_dir)
            if not success:
                print(" ⚠️ Frontend build failed. Backend will still work.")
            return success
        else:
            print(" ✅ Frontend already built.")
            return True

    # ====================================================

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
        print("   ./install.sh --force          Reinstall everything")
        print("   ./install.sh --skip-install   Rebuild frontend only (fast)")
        print("\n To start the app: ./start_services_local.sh")
        print("=" * 70 + "\n")
        return False

    def write_install_marker(self):
        """Write marker file indicating successful installation."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.install_marker, "w") as f:
                f.write(f"Linux local install completed: {timestamp}\n")
                f.write(f"Python: {self.python_cmd}\n")
            self.logger.info("Install marker written: %s", self.install_marker)
        except Exception as e:
            self.logger.warning("Could not write install marker: %s", e)

    def run_installation(self):
        self.print_banner()

        if not self.check_already_installed():
            return True

        if not self.verify_vendored_backend():
            return False
        self.check_aria2c()
        if not self.install_dependencies():
            return False
        self.install_pytorch_cuda121()
        self.apply_editable_installs()

        # =============== NEW: FRONTEND SETUP ===============
        # Always ensure frontend is ready, even with --skip-install
        if not self.install_frontend_deps():
            self.logger.warning("Frontend dependency installation failed.")
        if not self.build_frontend():
            self.logger.warning("Frontend build failed.")
        # ===================================================

        print("\n" + "=" * 70)
        print(" ✅ Local Linux Installation Complete!")
        print(f"    Log: {self.log_file}")
        print("")
        print(" 🚀 Next: Run ./start_services_local.sh")
        print("    - Backend: http://localhost:8000")
        print("    - Frontend: http://localhost:3000")
        print("=" * 70)
        self.write_install_marker()
        return True


def main():
    parser = argparse.ArgumentParser(description="Local Linux Installer (NVIDIA GPU)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install")
    parser.add_argument("--force", "-f", action="store_true", help="Force reinstall even if already installed")
    args = parser.parse_args()
    installer = LocalLinuxInstaller(verbose=args.verbose, skip_install=args.skip_install, force=args.force)
    success = installer.run_installation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
