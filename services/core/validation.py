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
    datasets_resolved = DATASETS_DIR.resolve()

    # Check if absolute path provided
    path_obj = Path(dataset_name)
    if path_obj.is_absolute():
        try:
            resolved = path_obj.resolve()
            # Use is_relative_to for proper cross-platform comparison.
            # str.startswith fails on Windows when drive letter casing differs
            # (e.g. "i:\" vs "I:\"), causing the check to fall through and strip
            # all backslashes from the Windows path — producing garbage like
            # "IKtiseos-Nyx-TrainerdatasetsBlueBra" that obviously doesn't exist.
            if resolved == datasets_resolved or resolved.is_relative_to(datasets_resolved):
                return resolved
        except (ValueError, OSError):
            pass  # Fall through to relative handling

    # Strip "datasets/" prefix if user included it (be forgiving!)
    if dataset_name.startswith("datasets/"):
        dataset_name = dataset_name[9:]
    elif dataset_name.startswith("datasets\\"):
        dataset_name = dataset_name[9:]

    # Remove path traversal attempts
    clean_name = dataset_name.replace("..", "").replace("/", "").replace("\\", "").strip()

    if not clean_name:
        raise ValidationError("Dataset name cannot be empty")

    # Construct path
    dataset_path = DATASETS_DIR / clean_name

    # Ensure it resolves within datasets directory
    try:
        resolved = dataset_path.resolve()
        if not (resolved == datasets_resolved or resolved.is_relative_to(datasets_resolved)):
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
    models_resolved = MODELS_DIR.resolve()
    vae_resolved = VAE_DIR.resolve()

    def _in_model_dirs(p: Path) -> bool:
        return p == models_resolved or p.is_relative_to(models_resolved) or \
               p == vae_resolved or p.is_relative_to(vae_resolved)

    # Check if absolute path provided
    path_obj = Path(model_name)
    if path_obj.is_absolute():
        try:
            resolved = path_obj.resolve()
            if _in_model_dirs(resolved):
                return resolved
        except (ValueError, OSError):
            pass

    # Remove path traversal attempts
    clean_name = model_name.replace("..", "").replace("/", "").replace("\\", "").strip()

    if not clean_name:
        raise ValidationError("Model name cannot be empty")

    # Construct path
    model_path = MODELS_DIR / clean_name

    try:
        resolved = model_path.resolve()
        if not _in_model_dirs(resolved):
            vae_check = (VAE_DIR / clean_name).resolve()
            if vae_check.exists() and _in_model_dirs(vae_check):
                return vae_check
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


def validate_image_path(image_path: str) -> Path:
    """
    Validate image path is within datasets directory.

    Args:
        image_path: Path to image file (e.g., "datasets/my_dataset/image.jpg" or "/full/path/to/image.jpg")

    Returns:
        Path: Validated absolute path to image

    Raises:
        ValidationError: If path traversal detected or path outside datasets

    Example:
        >>> path = validate_image_path("datasets/my_dataset/image.jpg")
        >>> str(path)
        '/path/to/datasets/my_dataset/image.jpg'
    """
    # Convert to Path and resolve to absolute
    try:
        img_path = Path(image_path).resolve()
    except (ValueError, OSError) as e:
        raise ValidationError(f"Invalid image path: {e}")

    # Ensure it's within DATASETS_DIR
    datasets_resolved = DATASETS_DIR.resolve()
    if not (img_path == datasets_resolved or img_path.is_relative_to(datasets_resolved)):
        raise ValidationError(f"Image path must be within datasets directory: {image_path}")

    # Check it has valid image extension
    if img_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValidationError(f"Invalid image extension: {img_path.suffix}. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}")

    return img_path


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
    output_resolved = OUTPUT_DIR.resolve()

    # Check if absolute path provided
    path_obj = Path(filename)
    if path_obj.is_absolute():
        try:
            resolved = path_obj.resolve()
            if resolved == output_resolved or resolved.is_relative_to(output_resolved):
                return resolved
        except (ValueError, OSError):
            pass

    clean_name = filename.replace("..", "").replace("/", "").replace("\\", "").strip()

    if not clean_name:
        raise ValidationError("Output filename cannot be empty")

    output_path = OUTPUT_DIR / clean_name

    try:
        resolved = output_path.resolve()
        if not (resolved == output_resolved or resolved.is_relative_to(output_resolved)):
            raise ValidationError(f"Invalid output path: {filename}")
    except (ValueError, OSError) as e:
        raise ValidationError(f"Invalid output path: {e}")

    return output_path


def validate_path_within(user_path: str, allowed_dirs: list) -> Path:
    """
    Generic path validation: ensure path is within one of the allowed directories.

    Args:
        user_path: User-provided path (absolute or relative)
        allowed_dirs: List of Path objects representing allowed base directories

    Returns:
        Path: Resolved, validated absolute path

    Raises:
        ValidationError: If path is outside all allowed directories
    """
    try:
        resolved = Path(user_path).resolve()
    except (ValueError, OSError) as e:
        raise ValidationError(f"Invalid path: {e}")

    for allowed_dir in allowed_dirs:
        allowed_resolved = Path(allowed_dir).resolve()
        # Check exact match or is a child (with separator to prevent prefix attacks)
        if resolved == allowed_resolved or resolved.is_relative_to(allowed_resolved):
            return resolved

    raise ValidationError("Access denied: path outside allowed directories")
