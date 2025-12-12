"""
Input validation utilities.

Prevents path traversal attacks and validates user inputs.
All paths are resolved and checked to ensure they're within allowed directories.
"""

from pathlib import Path
from typing import Set

from .exceptions import ValidationError

# Base directories (resolved to absolute paths)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATASETS_DIR = (PROJECT_ROOT / "datasets").resolve()
MODELS_DIR = (PROJECT_ROOT / "pretrained_model").resolve()
OUTPUT_DIR = (PROJECT_ROOT / "output").resolve()
VAE_DIR = (PROJECT_ROOT / "vae").resolve()

# Allowed image extensions
ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def validate_dataset_path(dataset_name: str) -> Path:
    """
    Validate dataset path is within datasets directory.

    Args:
        dataset_name: Name of the dataset (e.g., "my_character" or "datasets/my_character")

    Returns:
        Path: Validated absolute path to dataset

    Raises:
        ValidationError: If path traversal detected or invalid name

    Example:
        >>> path = validate_dataset_path("my_character")
        >>> str(path)
        '/path/to/datasets/my_character'
        >>> path = validate_dataset_path("datasets/my_character")
        >>> str(path)
        '/path/to/datasets/my_character'
    """
    # Strip "datasets/" prefix if user included it (be forgiving!)
    if dataset_name.startswith("datasets/"):
        dataset_name = dataset_name[9:]  # Remove "datasets/"
    elif dataset_name.startswith("datasets\\"):
        dataset_name = dataset_name[9:]  # Remove "datasets\"

    # Remove path traversal attempts
    clean_name = dataset_name.replace("..", "").replace("/", "").replace("\\", "").strip()

    if not clean_name:
        raise ValidationError("Dataset name cannot be empty")

    # Construct path
    dataset_path = DATASETS_DIR / clean_name

    # Ensure it resolves within datasets directory
    try:
        resolved = dataset_path.resolve()
        if not str(resolved).startswith(str(DATASETS_DIR)):
            raise ValidationError(f"Invalid dataset path: {dataset_name}")
    except (ValueError, OSError) as e:
        raise ValidationError(f"Invalid dataset path: {e}")

    return dataset_path


def validate_model_path(model_name: str) -> Path:
    """
    Validate model path is within models directory.

    Args:
        model_name: Name of the model file (e.g., "sdxl-base.safetensors")

    Returns:
        Path: Validated absolute path to model

    Raises:
        ValidationError: If path traversal detected or invalid name
    """
    # Remove path traversal attempts
    clean_name = model_name.replace("..", "").replace("/", "").replace("\\", "").strip()

    if not clean_name:
        raise ValidationError("Model name cannot be empty")

    # Construct path
    model_path = MODELS_DIR / clean_name

    # Ensure it resolves within models directory
    try:
        resolved = model_path.resolve()
        if not str(resolved).startswith(str(MODELS_DIR)):
            raise ValidationError(f"Invalid model path: {model_name}")
    except (ValueError, OSError) as e:
        raise ValidationError(f"Invalid model path: {e}")

    return model_path


def validate_image_filename(filename: str) -> str:
    """
    Validate image filename has allowed extension.

    Args:
        filename: Image filename (e.g., "image.jpg")

    Returns:
        str: Just the filename (no path components)

    Raises:
        ValidationError: If file type not allowed

    Example:
        >>> validate_image_filename("test.jpg")
        'test.jpg'
        >>> validate_image_filename("../hack.exe")
        ValidationError: Invalid file type: .exe
    """
    # Get just the filename (remove any path components)
    safe_name = Path(filename).name

    # Check extension
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValidationError(f"Invalid file type: {ext}. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}")

    return safe_name


def validate_output_path(filename: str) -> Path:
    """
    Validate output path is within output directory.

    Args:
        filename: Output filename (e.g., "my_lora.safetensors")

    Returns:
        Path: Validated absolute path

    Raises:
        ValidationError: If path traversal detected
    """
    clean_name = filename.replace("..", "").replace("/", "").replace("\\", "").strip()

    if not clean_name:
        raise ValidationError("Output filename cannot be empty")

    output_path = OUTPUT_DIR / clean_name

    try:
        resolved = output_path.resolve()
        if not str(resolved).startswith(str(OUTPUT_DIR)):
            raise ValidationError(f"Invalid output path: {filename}")
    except (ValueError, OSError) as e:
        raise ValidationError(f"Invalid output path: {e}")

    return output_path
