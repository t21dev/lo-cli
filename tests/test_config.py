"""Tests for configuration handling."""

from pathlib import Path
import tempfile

import pytest
import yaml

from locli.config import (
    LoCLIConfig,
    LoRAConfig,
    TrainingConfig,
    load_config,
    load_yaml_config,
    estimate_vram,
    get_default_config,
)


class TestLoRAConfig:
    """Tests for LoRAConfig."""

    def test_default_values(self):
        """Test default LoRA configuration values."""
        config = LoRAConfig()
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules

    def test_custom_values(self):
        """Test custom LoRA configuration values."""
        config = LoRAConfig(r=32, lora_alpha=64, lora_dropout=0.1)
        assert config.r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.1


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default training configuration values."""
        config = TrainingConfig()
        assert config.learning_rate == 2e-4
        assert config.batch_size == 4
        assert config.num_epochs == 3
        assert config.max_seq_length == 2048

    def test_custom_values(self):
        """Test custom training configuration values."""
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=8,
            num_epochs=5,
        )
        assert config.learning_rate == 1e-4
        assert config.batch_size == 8
        assert config.num_epochs == 5


class TestLoCLIConfig:
    """Tests for main LoCLI configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = get_default_config()
        assert config.lora.r == 16
        assert config.training.batch_size == 4
        assert config.hardware.device == "cuda"
        assert config.early_stopping.enabled is True

    def test_nested_config(self):
        """Test nested configuration."""
        config = LoCLIConfig(
            lora=LoRAConfig(r=32),
            training=TrainingConfig(num_epochs=5),
        )
        assert config.lora.r == 32
        assert config.training.num_epochs == 5


class TestLoadYamlConfig:
    """Tests for YAML config loading."""

    def test_load_valid_yaml(self):
        """Test loading a valid YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(
                {
                    "lora": {"r": 32, "lora_alpha": 64},
                    "training": {"num_epochs": 5},
                },
                f,
            )
            f.flush()

            config_dict = load_yaml_config(Path(f.name))
            assert config_dict["lora"]["r"] == 32
            assert config_dict["training"]["num_epochs"] == 5

    def test_load_empty_yaml(self):
        """Test loading an empty YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            f.flush()

            config_dict = load_yaml_config(Path(f.name))
            assert config_dict == {}


class TestEstimateVram:
    """Tests for VRAM estimation."""

    def test_qlora_7b(self):
        """Test QLoRA VRAM estimation for 7B model."""
        vram = estimate_vram(7, "qlora")
        assert 5 <= vram <= 8  # Should be around 6GB

    def test_lora_7b(self):
        """Test LoRA VRAM estimation for 7B model."""
        vram = estimate_vram(7, "lora")
        assert 12 <= vram <= 16  # Should be around 14GB

    def test_qlora_70b(self):
        """Test QLoRA VRAM estimation for 70B model."""
        vram = estimate_vram(70, "qlora")
        assert 20 <= vram <= 30  # Should be around 24GB

    def test_linear_scaling(self):
        """Test that VRAM scales roughly linearly with model size."""
        vram_7b = estimate_vram(7, "qlora")
        vram_14b = estimate_vram(14, "qlora")
        # 14B should need roughly 1.5-2.5x the VRAM of 7B
        assert 1.5 <= vram_14b / vram_7b <= 2.5
