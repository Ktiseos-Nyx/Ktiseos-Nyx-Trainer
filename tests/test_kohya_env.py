"""
Unit tests for KohyaTrainer._build_env — verifies credential injection.
"""
from unittest.mock import patch


def test_wandb_key_injected_into_env():
    """WANDB_API_KEY must appear in subprocess env when wandb_key is set."""
    from services.trainers.kohya import KohyaTrainer
    from services.models.training import TrainingConfig

    config = TrainingConfig(
        project_name="test",
        model_type="SD1.5",
        pretrained_model_name_or_path="/fake/model.safetensors",
        train_data_dir="/fake/data",
        output_dir="/fake/output",
        resolution=512,
        wandb_key="wbtest-secret-key",
    )
    trainer = KohyaTrainer(config)
    env = trainer._build_env()
    assert env.get("WANDB_API_KEY") == "wbtest-secret-key"


def test_wandb_key_absent_when_not_set():
    """WANDB_API_KEY must not appear in env when wandb_key is empty."""
    from services.trainers.kohya import KohyaTrainer
    from services.models.training import TrainingConfig

    config = TrainingConfig(
        project_name="test",
        model_type="SD1.5",
        pretrained_model_name_or_path="/fake/model.safetensors",
        train_data_dir="/fake/data",
        output_dir="/fake/output",
        resolution=512,
    )
    trainer = KohyaTrainer(config)
    env = trainer._build_env()
    assert "WANDB_API_KEY" not in env
