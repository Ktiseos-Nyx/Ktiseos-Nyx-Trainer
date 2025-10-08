# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# core/config_manager.py
import os
from typing import Dict, Optional

import toml


class ConfigManager:
    """
    Dead Simple Config Manager
    
    Job: Find TOML files and tell you if they're ready for training.
    That's it. No complex state management, no widget communication chaos.
    Just pure file-hunting logic.
    """

    def __init__(self, config_dir: str = None):
        # Default to training_configs directory in project root
        self.project_root = os.getcwd()
        self.config_dir = config_dir or os.path.join(self.project_root, "training_configs")

        # Required TOML files for training
        self.required_files = ["config.toml", "dataset.toml"]

    def files_ready(self) -> bool:
        """
        Dead simple check: Are all required TOML files present?
        
        Returns:
            bool: True if any TOML files exist (Kohya generates dynamic names)
        """
        return self._find_any_toml_files() is not None

    def get_config_paths(self) -> Optional[Dict[str, str]]:
        """
        Get paths to all TOML configuration files
        
        Returns:
            Dict with file names as keys and full paths as values, or None if files missing
        """
        return self._find_any_toml_files()

    def get_training_status(self) -> Dict[str, any]:
        """
        Get detailed status of configuration files
        
        Returns:
            Dict with status information for UI display
        """
        status = {
            "ready": False,
            "found_files": [],
            "missing_files": [],
            "config_dir": self.config_dir,
            "message": ""
        }

        # Check if config directory exists
        if not os.path.exists(self.config_dir):
            status["message"] = f"❌ Config directory not found: {self.config_dir}"
            status["missing_files"] = self.required_files.copy()
            return status

        # Hunt for each required file
        for filename in self.required_files:
            file_path = os.path.join(self.config_dir, filename)
            if os.path.exists(file_path):
                status["found_files"].append({
                    "name": filename,
                    "path": file_path,
                    "size": self._get_file_size(file_path),
                    "modified": self._get_file_modified(file_path)
                })
            else:
                status["missing_files"].append(filename)

        # Determine overall status
        if len(status["missing_files"]) == 0:
            status["ready"] = True
            status["message"] = f"✅ All config files found! Ready to train! ({len(status['found_files'])} files)"
        else:
            missing_list = ", ".join(status["missing_files"])
            status["message"] = f"❌ Missing files: {missing_list}"

        return status

    def validate_config_contents(self) -> Dict[str, any]:
        """
        Validate that TOML files have required content (optional deep check)
        
        Returns:
            Dict with validation results
        """
        validation = {
            "valid": False,
            "errors": [],
            "warnings": []
        }

        paths = self.get_config_paths()
        if not paths:
            validation["errors"].append("❌ Config files not found")
            return validation

        # Validate config.toml
        try:
            with open(paths["config.toml"], 'r') as f:
                config_data = toml.load(f)

            # Check for required sections
            required_sections = ["network_arguments", "optimizer_arguments", "training_arguments"]
            for section in required_sections:
                if section not in config_data:
                    validation["errors"].append(f"❌ Missing section in config.toml: {section}")

        except Exception as e:
            validation["errors"].append(f"❌ Error reading config.toml: {e}")

        # Validate dataset.toml
        try:
            with open(paths["dataset.toml"], 'r') as f:
                dataset_data = toml.load(f)

            # Check for required sections
            if "datasets" not in dataset_data:
                validation["errors"].append("❌ Missing 'datasets' section in dataset.toml")

        except Exception as e:
            validation["errors"].append(f"❌ Error reading dataset.toml: {e}")

        # Overall validation result
        validation["valid"] = len(validation["errors"]) == 0

        if validation["valid"]:
            validation["message"] = "✅ All config files are valid!"
        else:
            validation["message"] = f"❌ Validation failed ({len(validation['errors'])} errors)"

        return validation

    def _find_any_toml_files(self) -> Optional[Dict[str, str]]:
        """
        Internal method: Hunt for ANY TOML files (Kohya generates dynamic names)
        
        Returns:
            Dict of filename -> path, or None if no TOML files found
        """
        if not os.path.exists(self.config_dir):
            return None

        found_paths = {}

        # Find all TOML files in the config directory
        import glob
        toml_files = glob.glob(os.path.join(self.config_dir, "*.toml"))

        if not toml_files:
            return None  # No TOML files found

        # Add all found TOML files
        for file_path in toml_files:
            filename = os.path.basename(file_path)
            found_paths[filename] = file_path

        return found_paths

    def _find_all_toml_files(self) -> Optional[Dict[str, str]]:
        """
        Internal method: Hunt for all required TOML files
        
        Returns:
            Dict of filename -> path, or None if any files missing
        """
        if not os.path.exists(self.config_dir):
            return None

        found_paths = {}

        for filename in self.required_files:
            file_path = os.path.join(self.config_dir, filename)
            if os.path.exists(file_path):
                found_paths[filename] = file_path
            else:
                # Missing file = not ready
                return None

        return found_paths

    def _get_file_size(self, file_path: str) -> str:
        """Get human-readable file size"""
        try:
            size_bytes = os.path.getsize(file_path)
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
        except Exception:
            return "Unknown"

    def _get_file_modified(self, file_path: str) -> str:
        """Get human-readable file modification time"""
        try:
            import datetime
            mtime = os.path.getmtime(file_path)
            return datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return "Unknown"

    def cleanup_old_configs(self) -> Dict[str, any]:
        """
        Optional: Clean up old/stale configuration files
        
        Returns:
            Dict with cleanup results
        """
        cleanup_result = {
            "cleaned": False,
            "removed_files": [],
            "message": ""
        }

        if not os.path.exists(self.config_dir):
            cleanup_result["message"] = "📁 Config directory doesn't exist - nothing to clean"
            return cleanup_result

        try:
            # Find all TOML files in config directory
            all_files = [f for f in os.listdir(self.config_dir) if f.endswith('.toml')]

            for filename in all_files:
                file_path = os.path.join(self.config_dir, filename)
                os.remove(file_path)
                cleanup_result["removed_files"].append(filename)

            cleanup_result["cleaned"] = True
            if cleanup_result["removed_files"]:
                removed_count = len(cleanup_result["removed_files"])
                cleanup_result["message"] = f"🧹 Cleaned {removed_count} config files"
            else:
                cleanup_result["message"] = "✨ Config directory already clean"

        except Exception as e:
            cleanup_result["message"] = f"❌ Cleanup failed: {e}"

        return cleanup_result

    def print_status(self):
        """
        Convenience method: Print current config status to console
        Useful for debugging in Jupyter notebooks
        """
        print("🔍 CONFIG MANAGER STATUS")
        print("=" * 40)

        status = self.get_training_status()
        print(f"📁 Config Directory: {status['config_dir']}")
        print(f"📊 Status: {status['message']}")

        if status['found_files']:
            print("\n✅ Found Files:")
            for file_info in status['found_files']:
                print(f"   • {file_info['name']} ({file_info['size']})")

        if status['missing_files']:
            print("\n❌ Missing Files:")
            for filename in status['missing_files']:
                print(f"   • {filename}")

        print("\n🎯 Ready for Training:", "YES" if status['ready'] else "NO")
