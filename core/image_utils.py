# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# core/image_utils.py
import glob
import os

from PIL import ImageFile

# Fix for truncated JPEG images - allows PIL to handle corrupted/truncated files gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True

def count_images_in_directory(directory_path):
    """Count image files in a directory (recursive and non-recursive search)."""
    if not os.path.exists(directory_path):
        return 0

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.tiff']
    found_files = set()

    for ext in image_extensions:
        # Search recursively
        pattern_recursive = os.path.join(directory_path, '**', ext)
        for file in glob.glob(pattern_recursive, recursive=True):
            found_files.add(file)

        # Search non-recursively (for files directly in the specified directory)
        pattern_non_recursive = os.path.join(directory_path, ext)
        for file in glob.glob(pattern_non_recursive):
            found_files.add(file)

    return len(found_files)
