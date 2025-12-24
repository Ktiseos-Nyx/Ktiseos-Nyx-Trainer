#!/usr/bin/env python3
"""
Ktiseos-Nyx-Trainer - Local Linux Installer
For local development on Ubuntu/Debian/Fedora with NVIDIA GPU.
Installs CUDA 12.1 dependencies. Does NOT assume root or cloud environment.
"""

import argparse
import datetime
import logging
import os
import shutil
import subprocess
import sys


class LocalLinuxInstaller:
    def __init__(self, verbose=False, skip_install=False):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.verbose = verbose
        self.skip_install = skip_install
        self.setup_logging()
        self.python_cmd = sys.executable
        self.package_manager = {
            "name": "pip",
            "install_cmd": [self.python_cmd, "-m", "pip", "install"],
            "available": True
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
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
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
        req_file = os.path.join(self.project_root, "requirements.txt")
        if not os.path.exists(req_file):
            print(" ❌ requirements.txt not found")
            return False
        cmd = [self.python_cmd, "-m", "pip", "install", "-r", req_file]
        return self.run_command(cmd, "Installing Python dependencies")

    def install_pytorch_cuda121(self):
        if self.has_nvidia_gpu():
            print(" 📦 Installing PyTorch with CUDA 12.1...")
            cmd = [
                self.python_cmd, "-m", "pip", "install",
                "torch==2.4.0", "torchvision==0.19.0",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ]
            return self.run_command(cmd, "Installing PyTorch CUDA 12.1")
        else:
            print(" ⚠️ No NVIDIA GPU detected — skipping CUDA PyTorch")
            print("    Install manually if needed: pip install torch torchvision")
            return True

    def ensure_bitsandbytes_binaries(self):
        bits_dir = os.path.join(self.sd_scripts_dir, "bitsandbytes")
        required = ["libbitsandbytes_cuda121.so", "libbitsandbytes_cuda121_nocublaslt.so"]
        missing = [f for f in required if not os.path.exists(os.path.join(bits_dir, f))]
        if missing:
            print(f" ❌ Missing bitsandbytes binaries: {missing}")
            print("    Please ensure these are in trainer/derrian_backend/sd_scripts/bitsandbytes/")
            return False
        print(" ✅ bitsandbytes CUDA 12.1 binaries present")
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

    def run_installation(self):
        self.print_banner()
        if not self.verify_vendored_backend():
            return False
        self.check_aria2c()
        if not self.install_dependencies():
            return False
        self.install_pytorch_cuda121()
        if not self.ensure_bitsandbytes_binaries():
            return False
        self.apply_editable_installs()
        print("\n" + "=" * 70)
        print(" ✅ Local Linux Installation Complete!")
        print(f"    Log: {self.log_file}")
        print("")
        print(" 🚀 Next: Run ./start_services_local.sh")
        print("    - Backend: http://localhost:8000")
        print("    - Frontend: http://localhost:3000")
        print("=" * 70)
        return True


def main():
    parser = argparse.ArgumentParser(description="Local Linux Installer (NVIDIA GPU)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install")
    args = parser.parse_args()
    installer = LocalLinuxInstaller(verbose=args.verbose, skip_install=args.skip_install)
    success = installer.run_installation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
