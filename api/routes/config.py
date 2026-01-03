"""
Configuration API Routes
Handles training config templates, loading, and saving.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.models.training import TrainingConfig
from services.trainers.kohya_toml import KohyaTOMLGenerator

logger = logging.getLogger(__name__)
router = APIRouter()

# Project root detection
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class ConfigTemplate(BaseModel):
    """Configuration template"""
    name: str
    description: str
    config: Dict[str, Any]


class SaveConfigRequest(BaseModel):
    """Save configuration request"""
    name: str
    config: Dict[str, Any]


class ValidateConfigRequest(BaseModel):
    """Validate configuration request"""
    config: Dict[str, Any]


class ValidationResponse(BaseModel):
    """Validation response"""
    valid: bool
    errors: List[str] = []


@router.get("/templates")
async def list_config_templates():
    """List available configuration templates"""
    try:
        templates_dir = Path("example_configs")

        if not templates_dir.exists():
            return {"templates": []}

        templates = []
        for config_file in templates_dir.glob("*.toml"):
            try:
                config_data = toml.load(config_file)
                templates.append({
                    "name": config_file.stem,
                    "path": str(config_file),
                    "description": config_data.get("description", "No description")
                })
            except Exception as e:
                logger.warning(f"Failed to load template {config_file}: {e}")

        return {"templates": templates}

    except Exception as e:
        logger.error(f"Failed to list templates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/load")
async def load_config(path: str):
    """Load a configuration file"""
    try:
        config_path = Path(path)

        if not config_path.exists():
            raise HTTPException(status_code=404, detail="Config file not found")

        if config_path.suffix not in ['.toml', '.json']:
            raise HTTPException(
                status_code=400,
                detail="Only TOML and JSON configs supported"
            )

        # Load config
        if config_path.suffix == '.toml':
            config_data = toml.load(config_path)
        else:
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)

        return {
            "path": str(config_path),
            "config": config_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save")
async def save_config(request: SaveConfigRequest):
    """
    Save a configuration file (legacy single-file format).

    NOTE: For training configs, use /save-training instead to generate
    both dataset.toml and config.toml files that sd-scripts expects.
    """
    try:
        # Use relative path from project root
        config_dir = Path(__file__).parent.parent.parent / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        config_path = config_dir / f"{request.name}.toml"

        # Save as TOML
        with open(config_path, 'w') as f:
            toml.dump(request.config, f)

        logger.info(f"Saved config: {config_path}")

        return {
            "success": True,
            "path": str(config_path)
        }

    except Exception as e:
        logger.error(f"Failed to save config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-training")
async def save_training_config(config: TrainingConfig):
    """
    Save training configuration as TWO TOML files (dataset.toml + config.toml).

    This generates the same config files that sd-scripts expects for training,
    without actually starting the training process.

    Files are saved to:
    - config/{project_name}_dataset.toml
    - config/{project_name}_config.toml
    - trainer/runtime_store/{project_name}_dataset.toml (runtime copy)
    - trainer/runtime_store/{project_name}_config.toml (runtime copy)
    """
    try:
        # Initialize paths
        sd_scripts_dir = PROJECT_ROOT / "trainer" / "derrian_backend" / "sd_scripts"
        config_dir = PROJECT_ROOT / "config"
        runtime_dir = PROJECT_ROOT / "trainer" / "runtime_store"

        # Create directories
        config_dir.mkdir(parents=True, exist_ok=True)
        runtime_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TOML generator
        toml_generator = KohyaTOMLGenerator(
            config=config,
            project_root=PROJECT_ROOT,
            sd_scripts_dir=sd_scripts_dir
        )

        # Generate configs to config/ directory (user-facing)
        dataset_toml_path = config_dir / f"{config.project_name}_dataset.toml"
        config_toml_path = config_dir / f"{config.project_name}_config.toml"

        toml_generator.generate_dataset_toml(dataset_toml_path)
        toml_generator.generate_config_toml(config_toml_path)

        # Also save to runtime_store (used during training)
        runtime_dataset_path = runtime_dir / f"{config.project_name}_dataset.toml"
        runtime_config_path = runtime_dir / f"{config.project_name}_config.toml"

        toml_generator.generate_dataset_toml(runtime_dataset_path)
        toml_generator.generate_config_toml(runtime_config_path)

        logger.info(f"✅ Saved training configs for project: {config.project_name}")
        logger.info(f"   Dataset: {dataset_toml_path}")
        logger.info(f"   Config: {config_toml_path}")

        return {
            "success": True,
            "message": f"Training configs saved for {config.project_name}",
            "files": {
                "dataset": str(dataset_toml_path),
                "config": str(config_toml_path),
                "runtime_dataset": str(runtime_dataset_path),
                "runtime_config": str(runtime_config_path)
            }
        }

    except Exception as e:
        logger.error(f"Failed to save training config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=ValidationResponse)
async def validate_config(request: ValidateConfigRequest) -> ValidationResponse:
    """
    Validate a training configuration

    Checks for required fields and validates field types.
    Uses Pydantic for automatic validation.
    """
    try:
        config = request.config
        errors = []

        # Required fields validation
        required_fields = [
            "pretrained_model_name_or_path",
            "train_data_dir",
            "output_dir"
        ]

        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Type validation for common fields
        if "learning_rate" in config:
            try:
                float(config["learning_rate"])
            except (ValueError, TypeError):
                errors.append("learning_rate must be a number")

        if "max_train_epochs" in config:
            try:
                int(config["max_train_epochs"])
            except (ValueError, TypeError):
                errors.append("max_train_epochs must be an integer")

        if "train_batch_size" in config:
            try:
                batch_size = int(config["train_batch_size"])
                if batch_size < 1:
                    errors.append("train_batch_size must be >= 1")
            except (ValueError, TypeError):
                errors.append("train_batch_size must be an integer")

        return ValidationResponse(
            valid=len(errors) == 0,
            errors=errors
        )

    except Exception as e:
        logger.error(f"Failed to validate config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/defaults")
async def get_default_config():
    """Get default configuration values"""
    return {
        "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
        "resolution": 512,
        "train_batch_size": 1,
        "learning_rate": 1e-4,
        "max_train_steps": 1000,
        "network_module": "networks.lora",
        "network_dim": 32,
        "network_alpha": 32,
        "gradient_checkpointing": True,
        "mixed_precision": "fp16",
        "save_precision": "fp16",
        "optimizer_type": "AdamW8bit",
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 100
    }
