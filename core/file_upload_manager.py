#!/usr/bin/env python3
"""
File Upload Manager for LoRA Easy Training system.
Handles direct file uploads and ZIP extraction with proper error handling and logging.
"""

import os
import shutil
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class FileUploadManager:
    """
    Manages file uploads and ZIP extraction for dataset preparation.
    Provides clean separation between widget logic and file handling.
    """
    
    def __init__(self, base_path: str = "datasets"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
    def upload_images(self, files: List[Dict[str, Any]], destination_folder: str) -> Tuple[bool, str]:
        """
        Upload multiple image files to destination folder.
        
        Args:
            files: List of file objects from FileUpload widget
            destination_folder: Target directory path
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            dest_path = Path(destination_folder)
            if not dest_path.exists():
                logger.error(f"Destination folder does not exist: {destination_folder}")
                return False, f"Folder {destination_folder} does not exist"
                
            uploaded_count = 0
            skipped_count = 0
            
            for file_data in files:
                filename = file_data.get('name', 'unknown')
                content = file_data.get('content', b'')
                
                if not content:
                    logger.warning(f"Skipping empty file: {filename}")
                    skipped_count += 1
                    continue
                    
                # Validate file extension
                if not self._is_valid_image(filename):
                    logger.warning(f"Skipping non-image file: {filename}")
                    skipped_count += 1
                    continue
                    
                # Write file to destination
                file_path = dest_path / filename
                try:
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    uploaded_count += 1
                    logger.info(f"Uploaded: {filename} ({len(content)} bytes)")
                    
                except Exception as e:
                    logger.error(f"Failed to write {filename}: {e}")
                    skipped_count += 1
                    
            message = f"✅ Uploaded {uploaded_count} images"
            if skipped_count > 0:
                message += f", skipped {skipped_count} files"
                
            logger.info(f"Upload complete: {uploaded_count} uploaded, {skipped_count} skipped")
            return True, message
            
        except Exception as e:
            logger.error(f"Image upload failed: {e}")
            return False, f"Upload failed: {str(e)}"
            
    def upload_and_extract_zip(self, zip_file: Dict[str, Any], destination_folder: str) -> Tuple[bool, str]:
        """
        Upload and extract a ZIP file to destination folder.
        
        Args:
            zip_file: ZIP file object from FileUpload widget
            destination_folder: Target directory path
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            dest_path = Path(destination_folder)
            if not dest_path.exists():
                logger.error(f"Destination folder does not exist: {destination_folder}")
                return False, f"Folder {destination_folder} does not exist"
                
            filename = zip_file.get('name', 'unknown.zip')
            content = zip_file.get('content', b'')
            
            if not content:
                logger.error(f"ZIP file is empty: {filename}")
                return False, "ZIP file is empty"
                
            # Create temporary ZIP file
            temp_zip_path = dest_path / f"temp_{filename}"
            try:
                with open(temp_zip_path, 'wb') as f:
                    f.write(content)
                    
                logger.info(f"Saved temporary ZIP: {temp_zip_path} ({len(content)} bytes)")
                
                # Extract ZIP file
                extracted_count = self._extract_zip(temp_zip_path, dest_path)
                
                # Clean up temporary file
                temp_zip_path.unlink()
                
                message = f"✅ Extracted {extracted_count} files from {filename}"
                logger.info(f"ZIP extraction complete: {extracted_count} files")
                return True, message
                
            except Exception as e:
                # Clean up temp file if it exists
                if temp_zip_path.exists():
                    temp_zip_path.unlink()
                raise e
                
        except Exception as e:
            logger.error(f"ZIP upload and extraction failed: {e}")
            return False, f"ZIP extraction failed: {str(e)}"
            
    def _extract_zip(self, zip_path: Path, dest_path: Path) -> int:
        """
        Extract ZIP file contents to destination, filtering for valid images.
        
        Args:
            zip_path: Path to ZIP file
            dest_path: Destination directory
            
        Returns:
            Number of files extracted
        """
        extracted_count = 0
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                # Skip directories
                if file_info.is_dir():
                    continue
                    
                filename = os.path.basename(file_info.filename)
                
                # Skip hidden files and non-images
                if filename.startswith('.') or not self._is_valid_image(filename):
                    logger.debug(f"Skipping: {filename}")
                    continue
                    
                # Extract file with safe filename
                safe_filename = self._sanitize_filename(filename)
                output_path = dest_path / safe_filename
                
                try:
                    with zip_ref.open(file_info) as source:
                        with open(output_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                    extracted_count += 1
                    logger.debug(f"Extracted: {safe_filename}")
                    
                except Exception as e:
                    logger.warning(f"Failed to extract {filename}: {e}")
                    
        return extracted_count
        
    def _is_valid_image(self, filename: str) -> bool:
        """Check if filename has valid image extension."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.tif'}
        return Path(filename).suffix.lower() in valid_extensions
        
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe filesystem usage."""
        # Remove or replace problematic characters
        invalid_chars = '<>:"/\\|?*'
        sanitized = filename
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        return sanitized
        
    def validate_upload_destination(self, destination_path: str) -> Tuple[bool, str]:
        """
        Validate that upload destination exists and is writable.
        
        Args:
            destination_path: Path to validate
            
        Returns:
            Tuple of (valid: bool, message: str)
        """
        try:
            dest = Path(destination_path)
            
            if not dest.exists():
                return False, f"Destination folder does not exist: {destination_path}"
                
            if not dest.is_dir():
                return False, f"Destination is not a directory: {destination_path}"
                
            # Test write permissions
            test_file = dest / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                return False, f"Cannot write to destination: {e}"
                
            return True, "Destination is valid and writable"
            
        except Exception as e:
            return False, f"Validation error: {e}"
            
    def get_upload_stats(self, destination_path: str) -> Dict[str, Any]:
        """
        Get statistics about files in upload destination.
        
        Args:
            destination_path: Path to analyze
            
        Returns:
            Dictionary with file statistics
        """
        try:
            dest = Path(destination_path)
            if not dest.exists():
                return {"error": "Path does not exist"}
                
            image_files = []
            other_files = []
            total_size = 0
            
            for file_path in dest.iterdir():
                if file_path.is_file():
                    size = file_path.stat().st_size
                    total_size += size
                    
                    if self._is_valid_image(file_path.name):
                        image_files.append({
                            "name": file_path.name,
                            "size": size
                        })
                    else:
                        other_files.append({
                            "name": file_path.name,
                            "size": size
                        })
                        
            return {
                "image_count": len(image_files),
                "other_count": len(other_files),
                "total_size": total_size,
                "image_files": image_files,
                "other_files": other_files
            }
            
        except Exception as e:
            return {"error": str(e)}