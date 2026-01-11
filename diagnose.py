#!/usr/bin/env python3
"""
Ktiseos-Nyx-Trainer Diagnostic Tool
Collects comprehensive system information for troubleshooting installation issues.

Usage: python diagnose.py
Output: Creates diagnostics_TIMESTAMP.json and diagnostics_TIMESTAMP.txt
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
    """Run command and return stdout, stderr, and return code."""
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=10
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


def get_python_installations():
    """Detect all Python installations on the system."""
    pythons = {}

    # Current Python
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

    # Windows py launcher
    if platform.system() == "Windows":
        py_list = run_command(["py", "--list-paths"], shell=True)
        pythons["py_launcher_list"] = py_list.get("stdout", "Not available")

        py_3_path = run_command(["py", "-3", "-c", "import sys; print(sys.executable)"], shell=True)
        pythons["py_3_resolves_to"] = py_3_path.get("stdout", "Not available")

        py_version = run_command(["py", "-3", "--version"], shell=True)
        pythons["py_3_version"] = py_version.get("stdout", "Not available")

    # Check common Python commands
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


def check_git_state():
    """Check git repository state."""
    git_info = {}

    # Check if in git repo
    git_check = run_command(["git", "rev-parse", "--is-inside-work-tree"])
    git_info["is_git_repo"] = git_check.get("success", False)

    if git_info["is_git_repo"]:
        # Current branch
        branch = run_command(["git", "branch", "--show-current"])
        git_info["current_branch"] = branch.get("stdout", "Unknown")

        # Latest commit
        commit = run_command(["git", "log", "--oneline", "-1"])
        git_info["latest_commit"] = commit.get("stdout", "Unknown")

        # Remote URL
        remote = run_command(["git", "remote", "get-url", "origin"])
        git_info["remote_origin"] = remote.get("stdout", "Unknown")

        # Uncommitted changes
        status = run_command(["git", "status", "--porcelain"])
        git_info["has_uncommitted_changes"] = bool(status.get("stdout"))
        git_info["uncommitted_count"] = len(status.get("stdout", "").splitlines())

    return git_info


def check_installer_files():
    """Check which installer files exist and their contents."""
    installer_info = {}
    project_root = Path.cwd()

    # Check installer files
    installer_files = [
        "install.bat",
        "install.sh",
        "installer.py",
        "installer_windows_local.py",
        "installer_linux.py"
    ]

    for filename in installer_files:
        filepath = project_root / filename
        if filepath.exists():
            installer_info[filename] = {
                "exists": True,
                "size": filepath.stat().st_size,
                "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
            }

            # For .bat files, check what they call
            if filename == "install.bat":
                try:
                    content = filepath.read_text(encoding='utf-8', errors='ignore')
                    # Find what installer it calls
                    for line in content.splitlines():
                        if "installer" in line.lower() and ".py" in line and not line.strip().startswith("REM"):
                            installer_info[filename]["calls"] = line.strip()
                            break
                except Exception as e:
                    installer_info[filename]["read_error"] = str(e)
        else:
            installer_info[filename] = {"exists": False}

    return installer_info


def check_venv_state():
    """Check virtual environment state."""
    venv_info = {}
    project_root = Path.cwd()

    # Check for common venv names
    venv_names = [".venv", "venv", "env"]

    for venv_name in venv_names:
        venv_path = project_root / venv_name
        if venv_path.exists():
            venv_info[venv_name] = {
                "exists": True,
                "path": str(venv_path.absolute())
            }

            # Check Python in venv
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

    return venv_info


def check_cache_directories():
    """Check for cached Python files that could cause issues."""
    cache_info = {}
    project_root = Path.cwd()

    # Check for __pycache__ directories
    pycache_dirs = list(project_root.rglob("__pycache__"))
    cache_info["pycache_directories"] = {
        "count": len(pycache_dirs),
        "paths": [str(p.relative_to(project_root)) for p in pycache_dirs[:10]]  # First 10
    }

    # Check for .pyc files
    pyc_files = list(project_root.rglob("*.pyc"))
    cache_info["pyc_files"] = {
        "count": len(pyc_files),
        "sample": [str(p.relative_to(project_root)) for p in pyc_files[:10]]  # First 10
    }

    # Check pip cache (if accessible)
    if "LOCALAPPDATA" in os.environ:
        pip_cache = Path(os.environ["LOCALAPPDATA"]) / "pip" / "cache"
        cache_info["pip_cache"] = {
            "exists": pip_cache.exists(),
            "path": str(pip_cache) if pip_cache.exists() else None
        }

    return cache_info


def check_dependencies():
    """Check if critical dependencies are available."""
    deps = {}

    # Node.js and npm
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

    # Git
    git_version = run_command(["git", "--version"])
    deps["git"] = {
        "available": git_version.get("success", False),
        "version": git_version.get("stdout", "Not installed")
    }

    # aria2c
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
        "python_installations": get_python_installations(),
        "environment_variables": check_environment(),
        "git_state": check_git_state(),
        "installer_files": check_installer_files(),
        "venv_state": check_venv_state(),
        "cache_directories": check_cache_directories(),
        "dependencies": check_dependencies(),
        "disk_space": check_disk_space()
    }

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_filename = f"diagnostics_{timestamp}.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"✅ JSON report saved to: {json_filename}")

    # Save human-readable text
    txt_filename = f"diagnostics_{timestamp}.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Ktiseos-Nyx-Trainer Diagnostic Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {diagnostics['timestamp']}\n")
        f.write(f"Platform: {diagnostics['platform']['system']} {diagnostics['platform']['release']}\n")
        f.write(f"Working Directory: {diagnostics['working_directory']}\n\n")

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

        f.write("=" * 70 + "\n")
        f.write("INSTALLER FILES\n")
        f.write("=" * 70 + "\n")
        for filename, info in diagnostics['installer_files'].items():
            if info['exists']:
                f.write(f"✅ {filename}\n")
                if 'calls' in info:
                    f.write(f"   Calls: {info['calls']}\n")
            else:
                f.write(f"❌ {filename} (not found)\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("VIRTUAL ENVIRONMENTS\n")
        f.write("=" * 70 + "\n")
        for venv_name, info in diagnostics['venv_state'].items():
            if info['exists']:
                f.write(f"✅ {venv_name} exists\n")
                if 'python_version' in info:
                    f.write(f"   Python: {info['python_version']}\n")
            else:
                f.write(f"❌ {venv_name} (not found)\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("CACHED FILES (potential issues)\n")
        f.write("=" * 70 + "\n")
        cache = diagnostics['cache_directories']
        f.write(f"__pycache__ directories: {cache['pycache_directories']['count']}\n")
        f.write(f".pyc files: {cache['pyc_files']['count']}\n")
        if cache.get('pip_cache', {}).get('exists'):
            f.write(f"Pip cache exists at: {cache['pip_cache']['path']}\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("DEPENDENCIES\n")
        f.write("=" * 70 + "\n")
        for dep, info in diagnostics['dependencies'].items():
            status = "✅" if info['available'] else "❌"
            f.write(f"{status} {dep}: {info['version']}\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("DISK SPACE\n")
        f.write("=" * 70 + "\n")
        disk = diagnostics['disk_space']
        f.write(f"Drive/Mount: {disk.get('drive') or disk.get('mount')}\n")
        f.write(f"Total: {disk['total_gb']} GB\n")
        f.write(f"Used: {disk['used_gb']} GB ({disk['percent_used']}%)\n")
        f.write(f"Free: {disk['free_gb']} GB\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("CRITICAL ISSUES DETECTED\n")
        f.write("=" * 70 + "\n")

        issues = []

        # Check for Python version issues
        current_py = diagnostics['python_installations']['current']['version_info']
        if current_py['major'] < 3 or (current_py['major'] == 3 and current_py['minor'] < 10):
            issues.append(f"❌ Python {current_py['major']}.{current_py['minor']} is too old (requires 3.10+)")

        # Check for Microsoft Store Python
        if "WindowsApps" in diagnostics['python_installations']['current']['executable']:
            issues.append("⚠️ Using Microsoft Store Python (not recommended)")

        # Check for cached bytecode
        if cache['pycache_directories']['count'] > 0:
            issues.append(f"⚠️ {cache['pycache_directories']['count']} __pycache__ directories found (may cause issues)")

        # Check disk space
        if disk['free_gb'] < 50:
            issues.append(f"⚠️ Low disk space: {disk['free_gb']} GB free (50+ GB recommended)")

        # Check Node.js
        if not diagnostics['dependencies']['node']['available']:
            issues.append("❌ Node.js not installed (required for frontend)")

        # Check npm
        if not diagnostics['dependencies']['npm']['available']:
            issues.append("❌ npm not installed (required for frontend)")

        if issues:
            for issue in issues:
                f.write(f"{issue}\n")
        else:
            f.write("✅ No critical issues detected\n")

        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
        f.write("\nPlease attach this file to your GitHub issue for faster troubleshooting.\n")

    print(f"✅ Text report saved to: {txt_filename}")
    print()
    print("=" * 70)
    print("Diagnostic Complete!")
    print("=" * 70)
    print()
    print(f"📋 Please share {txt_filename} or {json_filename} when reporting issues.")
    print()

    # Show critical issues in console
    if "WindowsApps" in diagnostics['python_installations']['current']['executable']:
        print("⚠️  WARNING: You're using Microsoft Store Python!")
        print("   This can cause installation issues.")
        print("   Consider installing Python from python.org instead.")
        print()

    current_py = diagnostics['python_installations']['current']['version_info']
    if current_py['major'] < 3 or (current_py['major'] == 3 and current_py['minor'] < 10):
        print(f"❌ ERROR: Python {current_py['major']}.{current_py['minor']} is too old!")
        print("   This project requires Python 3.10 or newer.")
        print("   Download from: https://python.org/downloads/")
        print()


if __name__ == "__main__":
    main()
