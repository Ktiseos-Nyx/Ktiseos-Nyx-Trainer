#!/usr/bin/env python3
"""
Ktiseos-Nyx-Trainer - Frontend (Next.js) Installer

Handles Node.js version validation + auto-upgrade, npm install, and production
build. Designed for cloud GPU environments (RunPod, VastAI) where the base
image Node.js is often older than the project requirement.

Key behaviours:
  - Never exits on an old Node version — auto-downloads the correct LTS instead.
  - Idempotent: skips npm install if node_modules/ exists (unless --force).
  - Skips build if .next/ exists (unless --force).
  - Always logs to logs/frontend_install_<timestamp>.log.

Usage:
  python install_frontend.py               # Normal install + build
  python install_frontend.py --force       # Force reinstall + rebuild
  python install_frontend.py --skip-build  # Install deps only, no build
  python install_frontend.py -v            # Verbose output
"""

import argparse
import datetime
import logging
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

# Minimum Node.js version required by frontend/package.json
MIN_NODE_VERSION: Tuple[int, int, int] = (20, 19, 0)

# LTS version to install/upgrade to when current Node is missing or too old.
# Updated to 22.x (Jod LTS, active through April 2027).
NODE_INSTALL_VERSION = "v22.14.0"
NODE_INSTALL_URL = (
    f"https://nodejs.org/dist/{NODE_INSTALL_VERSION}/"
    f"node-{NODE_INSTALL_VERSION}-linux-x64.tar.xz"
)
NODE_INSTALL_DIR = "/usr/local"


class FrontendInstaller:
    """Installs and builds the Next.js frontend for cloud GPU environments."""

    def __init__(self, verbose: bool = False, force: bool = False, skip_build: bool = False):
        self.project_root = Path(__file__).parent.resolve()
        self.frontend_dir = self.project_root / "frontend"
        self.verbose = verbose
        self.force = force
        self.skip_build = skip_build
        self.setup_logging()

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #

    def setup_logging(self) -> None:
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Use the same daily log file as the app so all output is centralized
        datestamp = datetime.datetime.now().strftime("%Y%m%d")
        log_file = logs_dir / f"app_{datestamp}.log"

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        console_handler.setFormatter(formatter)

        self.logger = logging.getLogger("frontend_installer")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.log_file = log_file
        self.logger.info("Frontend installer started — log: %s", log_file)

    # ------------------------------------------------------------------ #
    #  Command runner                                                      #
    # ------------------------------------------------------------------ #

    def run_command(
        self,
        command: list,
        description: str,
        cwd: Optional[Path] = None,
        allow_failure: bool = False,
    ) -> bool:
        self.logger.info("%s...", description)
        self.logger.debug("Command: %s", " ".join(str(c) for c in command))
        if cwd:
            self.logger.debug("Working directory: %s", cwd)

        try:
            if self.verbose:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=cwd,
                )
                for line in iter(process.stdout.readline, ""):
                    line = line.rstrip()
                    if line:
                        print(f"   {line}")
                        self.logger.debug("OUTPUT: %s", line)
                process.stdout.close()
                returncode = process.wait()
                if returncode != 0:
                    raise subprocess.CalledProcessError(returncode, command)
            else:
                result = subprocess.run(
                    command, check=True, capture_output=True, text=True, cwd=cwd
                )
                self.logger.debug("stdout: %s", result.stdout[:500] if result.stdout else "")

            self.logger.info("%s successful.", description)
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error("%s failed (exit %s).", description, e.returncode)
            if hasattr(e, "stderr") and e.stderr:
                self.logger.error("stderr: %s", e.stderr[:500])
            if not allow_failure:
                return False
            self.logger.warning("Continuing despite failure (allow_failure=True).")
            return True

        except (OSError, subprocess.SubprocessError) as e:
            self.logger.error("Unexpected error during %s: %s", description, e)
            return False

    # ------------------------------------------------------------------ #
    #  Node.js detection and installation                                  #
    # ------------------------------------------------------------------ #

    def _discover_nvm_node(self) -> None:
        """
        Prepend the highest-version NVM-managed Node.js bin dir to PATH.

        RunPod and VastAI base images ship Node.js under /opt/nvm (or ~/.nvm)
        but do NOT add it to PATH automatically, so shutil.which("node") comes
        up empty even though Node is present.  Walk the same glob patterns the
        old bash provisioning scripts used and pick the newest version found.
        """
        import glob as _glob

        search_patterns = [
            "/opt/nvm/versions/node/*/bin",
            os.path.expanduser("~/.nvm/versions/node/*/bin"),
        ]

        candidates = []
        for pattern in search_patterns:
            for bin_dir in _glob.glob(pattern):
                node_bin = os.path.join(bin_dir, "node")
                if os.path.isfile(node_bin) and os.access(node_bin, os.X_OK):
                    candidates.append(bin_dir)

        if not candidates:
            return

        # Sort so we try the highest version last → prepend the best one
        candidates.sort()
        best = candidates[-1]
        self.logger.info("Found NVM Node.js at %s — adding to PATH", best)
        os.environ["PATH"] = best + os.pathsep + os.environ.get("PATH", "")

    def _parse_node_version(self) -> Optional[Tuple[int, int, int]]:
        """Return the active Node.js version tuple, or None if unavailable."""
        resolved = shutil.which("node")
        if not resolved:
            return None
        try:
            result = subprocess.run(
                [resolved, "--version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                return None
            version_str = result.stdout.strip().lstrip("v")
            parts = version_str.split(".")
            return (
                int(parts[0]),
                int(parts[1]),
                int(parts[2].split("-")[0]),
            )
        except (ValueError, IndexError, OSError, subprocess.SubprocessError):
            return None

    def _install_node_lts(self) -> bool:
        """
        Download and extract NODE_INSTALL_VERSION to NODE_INSTALL_DIR.

        Works on Linux x86-64 only (which covers all RunPod / VastAI images).
        Requires write access to NODE_INSTALL_DIR (root on cloud pods).
        """
        if platform.system().lower() != "linux":
            self.logger.error(
                "Auto Node.js install is only supported on Linux. "
                "Please install Node.js %s+ manually.",
                ".".join(str(v) for v in MIN_NODE_VERSION),
            )
            return False

        self.logger.info(
            "Downloading Node.js %s from %s", NODE_INSTALL_VERSION, NODE_INSTALL_URL
        )
        print(f" Downloading Node.js {NODE_INSTALL_VERSION}...")

        archive_path = Path("/tmp") / f"node-{NODE_INSTALL_VERSION}-linux-x64.tar.xz"
        try:
            urllib.request.urlretrieve(NODE_INSTALL_URL, archive_path)
        except OSError as e:
            self.logger.error("Failed to download Node.js: %s", e)
            return False

        self.logger.info("Extracting to %s ...", NODE_INSTALL_DIR)
        print(f" Extracting Node.js to {NODE_INSTALL_DIR}...")
        try:
            with tarfile.open(archive_path, "r:xz") as tf:
                # Strip the top-level node-vX.Y.Z-linux-x64/ directory
                install_root = Path(NODE_INSTALL_DIR).resolve()
                members = []
                for m in tf.getmembers():
                    parts = Path(m.name).parts
                    if len(parts) > 1:
                        m.name = str(Path(*parts[1:]))
                        # Skip entries that would escape the install directory
                        try:
                            (install_root / m.name).resolve().relative_to(install_root)
                        except ValueError:
                            self.logger.warning("Skipping unsafe tar entry: %s", m.name)
                            continue
                        members.append(m)
                tf.extractall(NODE_INSTALL_DIR, members=members)
        except (tarfile.TarError, OSError) as e:
            self.logger.error("Extraction failed: %s", e)
            return False
        finally:
            archive_path.unlink(missing_ok=True)

        # Update PATH to prefer the freshly installed Node binary
        node_bin = str(Path(NODE_INSTALL_DIR) / "bin")
        os.environ["PATH"] = node_bin + os.pathsep + os.environ.get("PATH", "")

        # Verify
        version = self._parse_node_version()
        if version and version >= MIN_NODE_VERSION:
            self.logger.info(
                "Node.js %d.%d.%d installed successfully.", *version
            )
            print(f" Node.js {'.'.join(str(v) for v in version)} ready.")
            return True

        self.logger.error(
            "Node.js installation appeared to succeed but version check failed."
        )
        return False

    def ensure_node(self) -> bool:
        """
        Ensure Node.js >= MIN_NODE_VERSION is available.

        On cloud pods the base image Node is typically too old (e.g. 18.x on
        RunPod). This method always upgrades rather than failing so provisioning
        can proceed unattended.
        """
        # Discover NVM-managed Node.js that isn't in PATH yet (RunPod / VastAI)
        self._discover_nvm_node()
        version = self._parse_node_version()

        if version is None:
            self.logger.warning(
                "Node.js not found — installing %s...", NODE_INSTALL_VERSION
            )
            print(f" Node.js not found — installing {NODE_INSTALL_VERSION}...")
            return self._install_node_lts()

        ver_str = "%d.%d.%d" % version
        if version >= MIN_NODE_VERSION:
            self.logger.info("Node.js %s meets requirement (>=%s).", ver_str,
                             ".".join(str(v) for v in MIN_NODE_VERSION))
            print(f" Node.js {ver_str} — OK")
            return True

        self.logger.warning(
            "Node.js %s is below the required %s — upgrading to %s...",
            ver_str,
            ".".join(str(v) for v in MIN_NODE_VERSION),
            NODE_INSTALL_VERSION,
        )
        print(
            f" Node.js {ver_str} is too old "
            f"(need >={'.'.join(str(v) for v in MIN_NODE_VERSION)}) "
            f"— upgrading to {NODE_INSTALL_VERSION}..."
        )
        return self._install_node_lts()

    # ------------------------------------------------------------------ #
    #  npm install                                                         #
    # ------------------------------------------------------------------ #

    def install_deps(self) -> bool:
        node_modules = self.frontend_dir / "node_modules"
        if node_modules.exists() and not self.force:
            self.logger.info(
                "node_modules/ exists — skipping npm install (use --force to reinstall)."
            )
            print(" node_modules/ already present — skipping npm install.")
            return True

        # Remove package-lock.json only when it was generated on a different OS.
        # CRLF line endings reliably indicate a Windows-generated lock file which
        # would cause Linux build failures (platform-specific native deps like
        # lightningcss resolve differently per OS).
        lock_file = self.frontend_dir / "package-lock.json"
        if lock_file.exists():
            try:
                is_foreign_platform = b"\r\n" in lock_file.read_bytes()
            except OSError:
                is_foreign_platform = False
            if is_foreign_platform:
                self.logger.info(
                    "Removing Windows-generated package-lock.json "
                    "(CRLF line endings detected — regenerating for this platform)."
                )
                lock_file.unlink()
            else:
                self.logger.info("package-lock.json matches current platform — keeping it.")

        npm = shutil.which("npm")
        if not npm:
            self.logger.error("npm not found after Node.js setup — cannot install deps.")
            return False

        # Try with --legacy-peer-deps first, fall back to --force
        for extra_flag in ["--legacy-peer-deps", "--legacy-peer-deps --force"]:
            flags = extra_flag.split()
            cmd = [npm, "install"] + flags
            self.logger.info("Running: %s", " ".join(cmd))
            success = self.run_command(
                cmd,
                f"npm install {extra_flag}",
                cwd=self.frontend_dir,
            )
            if success:
                return True
            self.logger.warning("npm install %s failed, trying next strategy...", extra_flag)

        self.logger.error("All npm install strategies failed.")
        return False

    # ------------------------------------------------------------------ #
    #  npm run build                                                       #
    # ------------------------------------------------------------------ #

    def build(self) -> bool:
        if self.skip_build:
            self.logger.info("Skipping build: --skip-build flag is set.")
            print(" Skipping build (--skip-build flag is set).")
            return True

        next_dir = self.frontend_dir / ".next"

        if next_dir.exists():
            # .next exists but --skip-build not set: still rebuild to pick up
            # any code changes since the last build.
            self.logger.info(".next/ exists — rebuilding.")

        npm = shutil.which("npm")
        if not npm:
            self.logger.error("npm not found — cannot build frontend.")
            return False

        print(" Building Next.js app (this takes ~30-60 seconds)...")
        success = self.run_command(
            [npm, "run", "build"],
            "npm run build",
            cwd=self.frontend_dir,
        )

        if success:
            self.logger.info("Frontend build complete.")
            print(" Frontend build complete.")
        else:
            self.logger.error(
                "Frontend build failed. Check %s for details.", self.log_file
            )
            print(f" Frontend build failed — see {self.log_file}")

        return success

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def run(self) -> bool:
        print("=" * 60)
        print("  Ktiseos-Nyx-Trainer — Frontend Setup")
        print("=" * 60)

        if not self.frontend_dir.exists():
            self.logger.error("frontend/ directory not found at %s", self.frontend_dir)
            print(f" ERROR: frontend/ not found at {self.frontend_dir}")
            return False

        if not self.ensure_node():
            self.logger.error("Node.js setup failed — frontend cannot be built.")
            print(" ERROR: Could not ensure a compatible Node.js version.")
            return False

        if not self.install_deps():
            self.logger.error("npm install failed.")
            print(" ERROR: npm install failed.")
            return False

        if not self.build():
            # Build failure is non-fatal for the provisioning flow — log and
            # continue so the backend still starts.
            self.logger.warning(
                "Frontend build failed — backend will still start, "
                "but the UI will be unavailable until rebuilt."
            )
            print(" WARNING: Frontend build failed — backend-only mode.")
            return False

        print("=" * 60)
        print("  Frontend setup complete!")
        print("=" * 60)
        return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ktiseos-Nyx-Trainer — Frontend installer for cloud GPU environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install_frontend.py               Normal install + build
  python install_frontend.py --force       Force reinstall even if already built
  python install_frontend.py --skip-build  Install npm deps only (no build)
  python install_frontend.py -v            Verbose output

Node.js:
  If Node.js is missing or below %(min_ver)s, the installer automatically
  downloads Node.js %(lts_ver)s to /usr/local. No manual intervention needed.
        """ % {
            "min_ver": ".".join(str(v) for v in MIN_NODE_VERSION),
            "lts_ver": NODE_INSTALL_VERSION,
        },
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force reinstall node_modules and rebuild .next even if they exist",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Run npm install but skip npm run build",
    )

    args = parser.parse_args()

    try:
        installer = FrontendInstaller(
            verbose=args.verbose,
            force=args.force,
            skip_build=args.skip_build,
        )
        success = installer.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
