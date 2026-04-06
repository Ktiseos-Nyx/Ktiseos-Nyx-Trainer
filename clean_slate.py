#!/usr/bin/env python3
"""
Clean Slate Reset Script for Ktiseos-Nyx-Trainer

Deletes build artifacts, virtual environments, and dependencies
to allow a fresh reinstall. SAFE: Preserves user data by default.

Usage:
    python clean_slate.py              # Interactive mode
    python clean_slate.py --yes        # Skip confirmation
    python clean_slate.py --nuclear    # Delete EVERYTHING (including models/datasets)
    python clean_slate.py --dry-run    # Preview what would be deleted
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# === Build artifacts & dependencies (always safe to nuke) ===
SAFE_DELETE_DIRS = [
    ".venv",
    "venv",
    "env",
    "node_modules",
    "frontend/node_modules",
    "frontend/.next",
    "frontend/build",
    "frontend/dist",
    "build",
    "dist",
    ".cache",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
]

# Glob patterns for scattered cache files
SAFE_DELETE_GLOBS = [
    "**/__pycache__",
    "logs/*.log",
    "*.log",
]

# Files inside runtime_store (but preserve .gitkeep)
RUNTIME_STORE = "trainer/runtime_store"

# === User data - ONLY with --nuclear ===
NUCLEAR_DIRS = [
    "pretrained_model",
    "vae",
    "tagger_models",
    "datasets",
    "output",
]


def confirm_action(message: str, auto_yes: bool) -> bool:
    if auto_yes:
        print(f"  [auto-yes] {message}")
        return True
    response = input(f"{message} [y/N]: ").strip().lower()
    return response in ("y", "yes")


def delete_path(path: Path, dry_run: bool = False) -> bool:
    """Safely delete a file or directory. Returns True if deleted."""
    if not path.exists():
        return False

    try:
        if dry_run:
            kind = "dir" if path.is_dir() else "file"
            print(f"  [DRY RUN] Would delete ({kind}): {path}")
            return True

        if path.is_file() or path.is_symlink():
            path.unlink()
        else:
            shutil.rmtree(path, ignore_errors=False)
        print(f"  Deleted: {path}")
        return True
    except PermissionError:
        print(f"  FAILED (permission denied): {path}")
        print(f"    -> Close any editors/terminals using this path and retry")
        return False
    except Exception as e:
        print(f"  FAILED: {path} ({e})")
        return False


def collect_paths(project_root: Path, nuclear: bool) -> list[Path]:
    """Build deduplicated, sorted list of paths to delete."""
    to_delete = []

    # Fixed directory paths
    for rel in SAFE_DELETE_DIRS:
        p = project_root / rel
        if p.exists():
            to_delete.append(p)

    # Glob patterns (skip vendored backend to avoid slow traversal)
    for pattern in SAFE_DELETE_GLOBS:
        for match in project_root.glob(pattern):
            # Don't descend into vendored backend for __pycache__ - too slow
            # We handle it separately below
            if "derrian_backend" not in str(match):
                to_delete.append(match)

    # Runtime store contents (preserve .gitkeep)
    rs = project_root / RUNTIME_STORE
    if rs.exists():
        for child in rs.iterdir():
            if child.name != ".gitkeep":
                to_delete.append(child)

    # Nuclear: user data dirs (preserve .gitkeep inside them)
    if nuclear:
        for rel in NUCLEAR_DIRS:
            d = project_root / rel
            if d.exists():
                for child in d.iterdir():
                    if child.name != ".gitkeep":
                        to_delete.append(child)

    # Deduplicate and sort
    return sorted(set(p for p in to_delete if p.exists()))


def main():
    parser = argparse.ArgumentParser(
        description="Clean Slate Reset for Ktiseos-Nyx-Trainer"
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompts"
    )
    parser.add_argument(
        "--nuclear",
        action="store_true",
        help="Also delete user data (models, VAEs, datasets, outputs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting",
    )
    args = parser.parse_args()

    # Script lives in project root
    project_root = Path(__file__).resolve().parent
    print(f"Project root: {project_root}\n")

    if args.nuclear:
        print(
            "!! NUCLEAR MODE: Models, VAEs, datasets, and outputs WILL be deleted !!\n"
        )
    else:
        print(
            "Safe mode: preserving models, VAEs, datasets, outputs."
        )
        print("Use --nuclear to delete those too.\n")

    to_delete = collect_paths(project_root, args.nuclear)

    if not to_delete:
        print("Nothing to clean - already fresh!")
        return 0

    print(f"Found {len(to_delete)} items to delete:\n")
    for p in to_delete:
        try:
            label = str(p.relative_to(project_root))
        except ValueError:
            label = str(p)
        kind = "dir" if p.is_dir() else "file"
        print(f"  [{kind}] {label}")
    print()

    if not args.dry_run and not confirm_action("Proceed with deletion?", args.yes):
        print("Cancelled. Nothing was deleted.")
        return 0

    print("Cleaning...\n")
    deleted = 0
    failed = 0
    for path in to_delete:
        if delete_path(path, dry_run=args.dry_run):
            deleted += 1
        else:
            failed += 1

    # Summary
    prefix = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{prefix}Done: {deleted} deleted, {failed} failed.")

    if not args.dry_run and failed == 0:
        if sys.platform == "win32":
            print("\nRun install.bat to reinstall fresh.")
        else:
            print("\nRun ./install.sh to reinstall fresh.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
