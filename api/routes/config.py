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

logger = logging.getLogger(__name__)
router = APIRouter()


class ConfigTemplate(BaseModel):
    """Configuration template"""
    name: str
    description: str
    config: Dict[str, Any]


class SaveConfigRequest(BaseModel):
    """Save configuration request"""
    name: str
    config: Dict[str, Any]


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
    """Save a configuration file"""
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


@router.get("/validate")
async def validate_config(config: Dict[str, Any]):
    """Validate a training configuration"""
    try:
        # TODO: Implement proper Pydantic validation for training configs
        # For now, basic validation
        required_fields = [
            "pretrained_model_name_or_path",
            "train_data_dir",
            "output_dir"
        ]

        missing = [f for f in required_fields if f not in config]

        if missing:
            return {
                "valid": False,
                "errors": [f"Missing required field: {f}" for f in missing]
            }

        return {
            "valid": True,
            "errors": []
        }

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
