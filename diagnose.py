#!/usr/bin/env python3
"""
Ktiseos-Nyx-Trainer Diagnostic Tool
Collects comprehensive system information for troubleshooting installation issues.

Usage: python diagnose.py
Output: Creates logs/diagnostics_TIMESTAMP.json and logs/diagnostics_TIMESTAMP.txt
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd, shell=False):
    """Run command and return dict with success, stdout, stderr, and returncode."""
    try:
        # On Windows, .cmd/.bat files need shell=True to execute
        if platform.system() == "Windows" and not shell:
            if isinstance(cmd, list) and cmd:
                # Try to resolve the executable via PATHEXT (finds npm.cmd, etc.)
                resolved = shutil.which(cmd[0])
                if resolved:
                    cmd = [resolved] + cmd[1:]
                else:
                    # Fall back to shell=True so cmd.exe handles .cmd resolution
                    shell = True

        if shell and isinstance(cmd, list):
            cmd = subprocess.list2cmdline([str(c) for c in cmd])

        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=30
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode
        }
    except FileNotFoundError:
        return {"success": False, "error": "Command not found", "stdout": "", "stderr": ""}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Command timed out", "stdout": "", "stderr": ""}
    except Exception as e:
        return {"success": False, "error": str(e), "stdout": "", "stderr": ""}


def get_python_installations():
    """Detect all Python installations on the system."""
    pythons = {}

    pythons["current"] = {
        "executable": sys.executable,
        "version": sys.version,
        "version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro
        },
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "is_venv": sys.prefix != sys.base_prefix
    }

    if platform.system() == "Windows":
        py_list = run_command(["py", "--list-paths"], shell=True)
        pythons["py_launcher_list"] = py_list.get("stdout", "Not available")

        py_3_path = run_command(["py", "-3", "-c", "import sys; print(sys.executable)"], shell=True)
        pythons["py_3_resolves_to"] = py_3_path.get("stdout", "Not available")

        py_version = run_command(["py", "-3", "--version"], shell=True)
        pythons["py_3_version"] = py_version.get("stdout", "Not available")

    for cmd in ["python", "python3", "python3.11", "python3.10"]:
        if shutil.which(cmd):
            result = run_command([cmd, "--version"])
            pythons[f"{cmd}_version"] = result.get("stdout", "Not available")

            exe_result = run_command([cmd, "-c", "import sys; print(sys.executable)"])
            pythons[f"{cmd}_executable"] = exe_result.get("stdout", "Not available")

    return pythons


def check_environment():
    """Check environment variables related to Python."""
    env_vars = {}
    important_vars = [
        "PATH", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
        "LOCALAPPDATA", "APPDATA", "TEMP", "TMP"
    ]

    for var in important_vars:
        env_vars[var] = os.environ.get(var, "Not set")

    return env_vars


def check_install_location():
    """Check if the project is installed in a problematic location."""
    project_root = Path.cwd()
    project_str = str(project_root)
    issues = []

    if platform.system() == "Windows":
        # Check for drive root installs (C:\ProjectName, D:\ProjectName)
        parts = project_root.parts
        if len(parts) <= 2:
            # Only flag as critical if on the OS drive (usually C:\)
            os_drive = os.environ.get("SystemDrive", "C:")
            if project_str.upper().startswith(os_drive.upper()):
                issues.append(
                    f"CRITICAL: Installed at OS drive root ({project_str}). "
                    "This can cause UAC permission errors. Move to a user folder like "
                    f"{os_drive}\\Users\\{os.environ.get('USERNAME', 'YourName')}\\Projects\\Ktiseos-Nyx-Trainer"
                )
            else:
                drive_letter = parts[0].rstrip("\\")
                issues.append(
                    f"INFO: Installed at drive root ({project_str}). "
                    "This is usually fine for non-OS drives. If you hit permission errors, "
                    "try a subfolder like "
                    f"{drive_letter}\\Projects\\Ktiseos-Nyx-Trainer"
                )

        # Check for Program Files
        prog_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        prog_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
        if project_str.startswith(prog_files) or project_str.startswith(prog_files_x86):
            issues.append(
                f"CRITICAL: Installed in Program Files ({project_str}). "
                "This location has restricted permissions. Move to a user folder."
            )

        # Check for Windows/System directories
        windir = os.environ.get("WINDIR", "C:\\Windows")
        if project_str.startswith(windir):
            issues.append(
                f"CRITICAL: Installed in Windows system directory ({project_str}). "
                "DO NOT install here. Move to a user folder immediately."
            )

        # Check for cloud sync folders
        cloud_indicators = ["OneDrive", "Dropbox", "Google Drive", "iCloudDrive"]
        for cloud in cloud_indicators:
            if cloud.lower() in project_str.lower():
                issues.append(
                    f"WARNING: Project appears to be in a {cloud} sync folder. "
                    "Cloud sync can cause file locking issues during installation. "
                    "Consider moving to a non-synced folder."
                )

    # Check write permissions
    test_file = project_root / "_write_test_diagnostic.tmp"
    try:
        test_file.write_text("test")
        test_file.unlink()
        writable = True
    except (PermissionError, OSError):
        writable = False
        issues.append(
            f"CRITICAL: Cannot write to project folder ({project_str}). "
            "Check folder permissions or move to a location you own."
        )

    return {
        "path": project_str,
        "writable": writable,
        "issues": issues
    }


def check_gpu():
    """Check NVIDIA GPU and CUDA availability."""
    gpu_info = {}

    # nvidia-smi
    smi = run_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    if smi.get("success"):
        gpu_info["nvidia_smi"] = smi["stdout"]
        gpu_info["gpu_available"] = True
    else:
        gpu_info["gpu_available"] = False
        gpu_info["nvidia_smi_error"] = smi.get("error") or smi.get("stderr") or "nvidia-smi not found"

    # Check CUDA via nvcc
    nvcc = run_command(["nvcc", "--version"])
    if nvcc.get("success"):
        gpu_info["nvcc_version"] = nvcc["stdout"].splitlines()[-1] if nvcc["stdout"] else "Unknown"
    else:
        gpu_info["nvcc_available"] = False

    # Check PyTorch CUDA
    torch_check = run_command([
        sys.executable, "-c",
        "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda}')"
    ])
    if torch_check.get("success"):
        gpu_info["pytorch_cuda"] = torch_check["stdout"]
    else:
        gpu_info["pytorch_cuda"] = "PyTorch not importable"

    return gpu_info


def check_key_packages():
    """Check if critical Python packages are importable and their versions."""
    packages = {}
    checks = [
        ("torch", "import torch; print(f'{torch.__version__} (CUDA: {torch.cuda.is_available()})')"),
        ("onnxruntime", "import onnxruntime as ort; print(f'{ort.__version__} - Providers: {ort.get_available_providers()}')"),
        ("diffusers", "import diffusers; print(diffusers.__version__)"),
        ("transformers", "import transformers; print(transformers.__version__)"),
        ("safetensors", "import safetensors; print(safetensors.__version__)"),
        ("lycoris_lora", "import lycoris; print('installed')"),
        ("bitsandbytes", "import bitsandbytes; print(bitsandbytes.__version__)"),
    ]

    for name, check_code in checks:
        result = run_command([sys.executable, "-c", check_code])
        if result.get("success"):
            packages[name] = {"installed": True, "info": result["stdout"]}
        else:
            packages[name] = {"installed": False, "error": result.get("stderr", result.get("error", "Import failed"))}

    return packages


def check_git_state():
    """Check git repository state."""
    git_info = {}

    git_check = run_command(["git", "rev-parse", "--is-inside-work-tree"])
    git_info["is_git_repo"] = git_check.get("success", False)

    if git_info["is_git_repo"]:
        branch = run_command(["git", "branch", "--show-current"])
        git_info["current_branch"] = branch.get("stdout", "Unknown")

        commit = run_command(["git", "log", "--oneline", "-1"])
        git_info["latest_commit"] = commit.get("stdout", "Unknown")

        remote = run_command(["git", "remote", "get-url", "origin"])
        git_info["remote_origin"] = remote.get("stdout", "Unknown")

        status = run_command(["git", "status", "--porcelain"])
        git_info["has_uncommitted_changes"] = bool(status.get("stdout"))
        git_info["uncommitted_count"] = len(status.get("stdout", "").splitlines())

    return git_info


def check_installer_files():
    """Check which installer files exist and their contents."""
    installer_info = {}
    project_root = Path.cwd()

    installer_files = [
        "install.bat",
        "install.sh",
        "installer.py",
        "installer_windows_local.py",
    ]

    for filename in installer_files:
        filepath = project_root / filename
        if filepath.exists():
            installer_info[filename] = {
                "exists": True,
                "size": filepath.stat().st_size,
                "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
            }

            if filename == "install.bat":
                try:
                    content = filepath.read_text(encoding='utf-8', errors='ignore')
                    for line in content.splitlines():
                        if "installer" in line.lower() and ".py" in line and not line.strip().startswith("REM"):
                            installer_info[filename]["calls"] = line.strip()
                            break
                except Exception as e:
                    installer_info[filename]["read_error"] = str(e)
        else:
            installer_info[filename] = {"exists": False}

    # Check for install marker
    marker = project_root / "install_complete.marker"
    if marker.exists():
        try:
            installer_info[".install_complete"] = {
                "exists": True,
                "content": marker.read_text(encoding='utf-8', errors='ignore').strip()
            }
        except Exception:
            installer_info[".install_complete"] = {"exists": True, "content": "unreadable"}
    else:
        installer_info[".install_complete"] = {"exists": False}

    return installer_info


def check_venv_state():
    """Check virtual environment state."""
    venv_info = {}
    project_root = Path.cwd()

    venv_names = [".venv", "venv", "env"]

    for venv_name in venv_names:
        venv_path = project_root / venv_name
        if venv_path.exists():
            venv_info[venv_name] = {
                "exists": True,
                "path": str(venv_path.absolute())
            }

            if platform.system() == "Windows":
                venv_python = venv_path / "Scripts" / "python.exe"
            else:
                venv_python = venv_path / "bin" / "python"

            if venv_python.exists():
                venv_version = run_command([str(venv_python), "--version"])
                venv_info[venv_name]["python_version"] = venv_version.get("stdout", "Unknown")

                venv_exe = run_command([str(venv_python), "-c", "import sys; print(sys.executable)"])
                venv_info[venv_name]["python_executable"] = venv_exe.get("stdout", "Unknown")
            else:
                venv_info[venv_name]["python_exists"] = False
        else:
            venv_info[venv_name] = {"exists": False}

    # Check if currently running inside a venv
    venv_info["currently_in_venv"] = sys.prefix != sys.base_prefix

    return venv_info


def check_dependencies():
    """Check if critical dependencies are available."""
    deps = {}

    node_version = run_command(["node", "--version"])
    deps["node"] = {
        "available": node_version.get("success", False),
        "version": node_version.get("stdout", "Not installed")
    }

    npm_version = run_command(["npm", "--version"])
    deps["npm"] = {
        "available": npm_version.get("success", False),
        "version": npm_version.get("stdout", "Not installed")
    }

    git_version = run_command(["git", "--version"])
    deps["git"] = {
        "available": git_version.get("success", False),
        "version": git_version.get("stdout", "Not installed")
    }

    aria2_version = run_command(["aria2c", "--version"])
    deps["aria2c"] = {
        "available": aria2_version.get("success", False),
        "version": aria2_version.get("stdout", "Not installed").splitlines()[0] if aria2_version.get("success") else "Not installed"
    }

    return deps


def check_disk_space():
    """Check available disk space."""
    project_root = Path.cwd()

    if platform.system() == "Windows":
        drive = Path(project_root.anchor)
        usage = shutil.disk_usage(drive)
        return {
            "drive": str(drive),
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "percent_used": round(usage.used / usage.total * 100, 2)
        }
    else:
        usage = shutil.disk_usage("/")
        return {
            "mount": "/",
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "percent_used": round(usage.used / usage.total * 100, 2)
        }


def check_frontend_state():
    """Check frontend build state."""
    project_root = Path.cwd()
    frontend_dir = project_root / "frontend"
    state = {}

    if not frontend_dir.exists():
        state["frontend_dir"] = False
        return state

    state["frontend_dir"] = True
    state["node_modules"] = (frontend_dir / "node_modules").exists()
    state["next_build"] = (frontend_dir / ".next").exists()
    state["package_json"] = (frontend_dir / "package.json").exists()

    if state["node_modules"]:
        # Check if next is actually in node_modules
        state["next_installed"] = (frontend_dir / "node_modules" / "next").exists()
    else:
        state["next_installed"] = False

    return state


def analyze_issues(diagnostics):
    """Analyze collected diagnostics and generate actionable issue list."""
    issues = []

    # Install location problems
    location = diagnostics.get("install_location", {})
    issues.extend(location.get("issues", []))

    # Python version
    current_py = diagnostics['python_installations']['current']['version_info']
    if current_py['major'] < 3 or (current_py['major'] == 3 and current_py['minor'] < 10):
        issues.append(f"CRITICAL: Python {current_py['major']}.{current_py['minor']} is too old (requires 3.10+)")

    # Microsoft Store Python
    if "WindowsApps" in diagnostics['python_installations']['current']['executable']:
        issues.append(
            "WARNING: Using Microsoft Store Python. This frequently causes PATH conflicts "
            "and permission issues. Install from python.org instead."
        )

    # Multiple Python installations pointing to different places
    py_exes = set()
    for key, val in diagnostics['python_installations'].items():
        if key.endswith("_executable") and isinstance(val, str) and val != "Not available":
            py_exes.add(val.strip().lower())
    if len(py_exes) > 1:
        issues.append(
            f"WARNING: Multiple Python installations detected ({len(py_exes)} different executables). "
            "This can cause confusion about which Python has your packages installed. "
            "Use a virtual environment to avoid conflicts."
        )

    # GPU
    gpu = diagnostics.get("gpu", {})
    if not gpu.get("gpu_available"):
        issues.append("WARNING: No NVIDIA GPU detected. Training requires a CUDA-capable GPU.")
    if "CUDA available: False" in gpu.get("pytorch_cuda", ""):
        issues.append(
            "CRITICAL: PyTorch is installed but cannot see CUDA. "
            "Reinstall PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )

    # Key packages
    packages = diagnostics.get("key_packages", {})
    if not packages.get("torch", {}).get("installed"):
        issues.append("CRITICAL: PyTorch is not installed. Run the installer.")
    if not packages.get("onnxruntime", {}).get("installed"):
        issues.append("WARNING: ONNX Runtime not installed. WD14 tagging will not work.")
    if not packages.get("lycoris_lora", {}).get("installed"):
        issues.append("WARNING: LyCORIS not installed. LoHa/LoKr/LoCon training will not work.")

    # Disk space
    disk = diagnostics['disk_space']
    if disk['free_gb'] < 20:
        issues.append(f"CRITICAL: Very low disk space: {disk['free_gb']} GB free. Need at least 20 GB.")
    elif disk['free_gb'] < 50:
        issues.append(f"WARNING: Low disk space: {disk['free_gb']} GB free (50+ GB recommended for models).")

    # Node.js
    if not diagnostics['dependencies']['node']['available']:
        issues.append("CRITICAL: Node.js not installed (required for the web UI frontend).")
    if not diagnostics['dependencies']['npm']['available']:
        issues.append("CRITICAL: npm not installed (required for frontend dependency installation).")

    # Frontend state
    frontend = diagnostics.get("frontend_state", {})
    if frontend.get("frontend_dir") and not frontend.get("node_modules"):
        issues.append("WARNING: Frontend node_modules missing. Run the installer or 'npm install' in frontend/.")
    if frontend.get("frontend_dir") and not frontend.get("next_build"):
        issues.append("WARNING: Frontend not built (.next/ missing). Run 'npm run build' in frontend/.")
    if frontend.get("node_modules") and not frontend.get("next_installed"):
        issues.append("WARNING: node_modules exists but 'next' package is missing. Try deleting node_modules and reinstalling.")

    # Venv state
    venv = diagnostics.get("venv_state", {})
    if not venv.get("currently_in_venv"):
        has_venv = any(v.get("exists") for k, v in venv.items() if isinstance(v, dict) and k != "currently_in_venv")
        if has_venv:
            issues.append(
                "WARNING: A virtual environment exists but you're NOT running inside it. "
                "Activate it first: .venv\\Scripts\\activate (Windows) or source .venv/bin/activate (Linux)"
            )

    return issues


def main():
    """Main diagnostic routine."""
    print("=" * 70)
    print("Ktiseos-Nyx-Trainer Diagnostic Tool")
    print("=" * 70)
    print()
    print("Collecting system information...")
    print()

    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        },
        "working_directory": str(Path.cwd()),
        "install_location": check_install_location(),
        "python_installations": get_python_installations(),
        "environment_variables": check_environment(),
        "gpu": check_gpu(),
        "key_packages": check_key_packages(),
        "git_state": check_git_state(),
        "installer_files": check_installer_files(),
        "venv_state": check_venv_state(),
        "dependencies": check_dependencies(),
        "disk_space": check_disk_space(),
        "frontend_state": check_frontend_state(),
    }

    # Analyze all collected data for issues
    issues = analyze_issues(diagnostics)
    diagnostics["issues_detected"] = issues

    # Output to logs/ directory instead of project root
    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_filename = logs_dir / f"diagnostics_{timestamp}.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"JSON report saved to: {json_filename}")

    # Save human-readable text
    txt_filename = logs_dir / f"diagnostics_{timestamp}.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Ktiseos-Nyx-Trainer Diagnostic Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {diagnostics['timestamp']}\n")
        f.write(f"Platform: {diagnostics['platform']['system']} {diagnostics['platform']['release']}\n")
        f.write(f"Working Directory: {diagnostics['working_directory']}\n\n")

        # Install location
        f.write("=" * 70 + "\n")
        f.write("INSTALL LOCATION\n")
        f.write("=" * 70 + "\n")
        loc = diagnostics['install_location']
        f.write(f"Path: {loc['path']}\n")
        f.write(f"Writable: {loc['writable']}\n")
        if loc['issues']:
            for issue in loc['issues']:
                f.write(f"  !! {issue}\n")
        f.write("\n")

        # Python
        f.write("=" * 70 + "\n")
        f.write("PYTHON INSTALLATIONS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Current Python: {diagnostics['python_installations']['current']['executable']}\n")
        f.write(f"Version: {diagnostics['python_installations']['current']['version']}\n")
        f.write(f"In Virtual Env: {diagnostics['python_installations']['current']['is_venv']}\n\n")

        if "py_launcher_list" in diagnostics['python_installations']:
            f.write("Windows py launcher output:\n")
            f.write(diagnostics['python_installations']['py_launcher_list'] + "\n\n")
            f.write(f"py -3 resolves to: {diagnostics['python_installations'].get('py_3_resolves_to', 'Unknown')}\n\n")

        # GPU
        f.write("=" * 70 + "\n")
        f.write("GPU & CUDA\n")
        f.write("=" * 70 + "\n")
        gpu = diagnostics['gpu']
        if gpu.get("gpu_available"):
            f.write(f"GPU: {gpu.get('nvidia_smi', 'Unknown')}\n")
        else:
            f.write(f"GPU: Not detected ({gpu.get('nvidia_smi_error', 'Unknown')})\n")
        f.write(f"PyTorch CUDA: {gpu.get('pytorch_cuda', 'Unknown')}\n")
        f.write("\n")

        # Key packages
        f.write("=" * 70 + "\n")
        f.write("KEY PACKAGES\n")
        f.write("=" * 70 + "\n")
        for pkg, info in diagnostics['key_packages'].items():
            if info['installed']:
                f.write(f"  OK {pkg}: {info['info']}\n")
            else:
                f.write(f"  MISSING {pkg}: {info.get('error', 'Not installed')}\n")
        f.write("\n")

        # Git
        f.write("=" * 70 + "\n")
        f.write("GIT STATE\n")
        f.write("=" * 70 + "\n")
        git = diagnostics['git_state']
        f.write(f"Is Git Repo: {git.get('is_git_repo', False)}\n")
        if git.get('is_git_repo'):
            f.write(f"Branch: {git.get('current_branch', 'Unknown')}\n")
            f.write(f"Latest Commit: {git.get('latest_commit', 'Unknown')}\n")
            f.write(f"Remote: {git.get('remote_origin', 'Unknown')}\n")
            f.write(f"Uncommitted Changes: {git.get('uncommitted_count', 0)} files\n")
        f.write("\n")

        # Installer files
        f.write("=" * 70 + "\n")
        f.write("INSTALLER FILES\n")
        f.write("=" * 70 + "\n")
        for filename, info in diagnostics['installer_files'].items():
            if isinstance(info, dict) and info.get('exists'):
                f.write(f"  OK {filename}\n")
                if 'calls' in info:
                    f.write(f"     Calls: {info['calls']}\n")
                if 'content' in info:
                    f.write(f"     Content: {info['content']}\n")
            else:
                f.write(f"  -- {filename} (not found)\n")
        f.write("\n")

        # Virtual environments
        f.write("=" * 70 + "\n")
        f.write("VIRTUAL ENVIRONMENTS\n")
        f.write("=" * 70 + "\n")
        for venv_name, info in diagnostics['venv_state'].items():
            if venv_name == "currently_in_venv":
                f.write(f"Currently in venv: {info}\n")
                continue
            if info.get('exists'):
                f.write(f"  OK {venv_name} exists\n")
                if 'python_version' in info:
                    f.write(f"     Python: {info['python_version']}\n")
            else:
                f.write(f"  -- {venv_name} (not found)\n")
        f.write("\n")

        # Frontend
        f.write("=" * 70 + "\n")
        f.write("FRONTEND STATE\n")
        f.write("=" * 70 + "\n")
        frontend = diagnostics['frontend_state']
        f.write(f"frontend/ directory: {'exists' if frontend.get('frontend_dir') else 'MISSING'}\n")
        if frontend.get('frontend_dir'):
            f.write(f"node_modules: {'exists' if frontend.get('node_modules') else 'MISSING'}\n")
            f.write(f"next package: {'installed' if frontend.get('next_installed') else 'MISSING'}\n")
            f.write(f".next build: {'exists' if frontend.get('next_build') else 'MISSING'}\n")
        f.write("\n")

        # Dependencies
        f.write("=" * 70 + "\n")
        f.write("SYSTEM DEPENDENCIES\n")
        f.write("=" * 70 + "\n")
        for dep, info in diagnostics['dependencies'].items():
            status = "OK" if info['available'] else "MISSING"
            f.write(f"  {status} {dep}: {info['version']}\n")
        f.write("\n")

        # Disk space
        f.write("=" * 70 + "\n")
        f.write("DISK SPACE\n")
        f.write("=" * 70 + "\n")
        disk = diagnostics['disk_space']
        f.write(f"Drive/Mount: {disk.get('drive') or disk.get('mount')}\n")
        f.write(f"Total: {disk['total_gb']} GB\n")
        f.write(f"Used: {disk['used_gb']} GB ({disk['percent_used']}%)\n")
        f.write(f"Free: {disk['free_gb']} GB\n")
        f.write("\n")

        # Issues summary
        f.write("=" * 70 + "\n")
        f.write("ISSUES DETECTED\n")
        f.write("=" * 70 + "\n")
        if issues:
            for issue in issues:
                f.write(f"  !! {issue}\n\n")
        else:
            f.write("  No issues detected.\n")

        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
        f.write("\nPlease attach this file to your GitHub issue for faster troubleshooting.\n")

    print(f"Text report saved to: {txt_filename}")
    print()
    print("=" * 70)
    print("Diagnostic Complete!")
    print("=" * 70)
    print()

    # Print issues to console
    if issues:
        print(f"Found {len(issues)} issue(s):\n")
        for issue in issues:
            print(f"  !! {issue}\n")
    else:
        print("No issues detected.\n")

    print(f"Share {txt_filename} when reporting issues on GitHub.")
    print()


if __name__ == "__main__":
    main()
