"""Configuration handling for LoCLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoRAConfig(BaseModel):
    """LoRA/QLoRA configuration."""

    r: int = Field(default=16, description="LoRA rank")
    lora_alpha: int = Field(default=32, description="LoRA alpha scaling factor")
    lora_dropout: float = Field(default=0.05, description="LoRA dropout rate")
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Target modules for LoRA",
    )
    bias: str = Field(default="none", description="Bias training mode")
    task_type: str = Field(default="CAUSAL_LM", description="Task type for PEFT")


class TrainingConfig(BaseModel):
    """Training configuration."""

    learning_rate: float = Field(default=2e-4, description="Learning rate")
    batch_size: int = Field(default=4, description="Batch size per device")
    gradient_accumulation_steps: int = Field(
        default=4, description="Gradient accumulation steps"
    )
    num_epochs: int = Field(default=3, description="Number of training epochs")
    warmup_ratio: float = Field(default=0.03, description="Warmup ratio")
    max_seq_length: int = Field(default=2048, description="Maximum sequence length")
    save_steps: int = Field(default=100, description="Save checkpoint every N steps")
    logging_steps: int = Field(default=10, description="Log every N steps")
    weight_decay: float = Field(default=0.01, description="Weight decay")
    max_grad_norm: float = Field(default=1.0, description="Max gradient norm for clipping")


class HardwareConfig(BaseModel):
    """Hardware configuration."""

    device: str = Field(default="cuda", description="Device to use (cuda, cpu)")
    bf16: bool = Field(default=True, description="Use bfloat16 precision")
    fp16: bool = Field(default=False, description="Use float16 precision")
    gradient_checkpointing: bool = Field(
        default=True, description="Enable gradient checkpointing"
    )


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""

    enabled: bool = Field(default=True, description="Enable early stopping")
    patience: int = Field(default=3, description="Patience for early stopping")
    min_delta: float = Field(default=0.001, description="Minimum delta for improvement")


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: str | None = Field(default=None, description="OpenAI API key")
    model: str = Field(default="gpt-4.1-mini", description="Model to use for suggestions")


class HuggingFaceConfig(BaseModel):
    """HuggingFace configuration."""

    token: str | None = Field(default=None, description="HuggingFace token")
    cache_dir: str | None = Field(default=None, description="Cache directory")


class LoCLIConfig(BaseModel):
    """Main LoCLI configuration."""

    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)


class EnvSettings(BaseSettings):
    """Environment variable settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str | None = Field(default=None, alias="OPENAI_MODEL")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    cuda_visible_devices: str | None = Field(default=None, alias="CUDA_VISIBLE_DEVICES")
    default_method: Literal["lora", "qlora"] = Field(default="qlora", alias="DEFAULT_METHOD")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    hf_home: str | None = Field(default=None, alias="HF_HOME")


def find_config_file() -> Path | None:
    """Find the configuration file in common locations."""
    search_paths = [
        Path.cwd() / "locli.yaml",
        Path.cwd() / "locli.yml",
        Path.cwd() / ".locli.yaml",
        Path.cwd() / ".locli.yml",
        Path.home() / ".config" / "locli" / "config.yaml",
        Path.home() / ".locli.yaml",
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


def load_yaml_config(path: Path) -> dict:
    """Load configuration from a YAML file."""
    with open(path) as f:
        content = f.read()

    # Expand environment variables in the YAML content
    for key, value in os.environ.items():
        content = content.replace(f"${{{key}}}", value)
        content = content.replace(f"${key}", value)

    return yaml.safe_load(content) or {}


def load_config(config_path: Path | None = None) -> LoCLIConfig:
    """Load LoCLI configuration from file and environment."""
    # Load environment settings first
    env_settings = EnvSettings()

    # Start with default config
    config_dict: dict = {}

    # Load from YAML file if available
    yaml_path = config_path or find_config_file()
    if yaml_path and yaml_path.exists():
        config_dict = load_yaml_config(yaml_path)

    # Create config from YAML
    config = LoCLIConfig(**config_dict)

    # Override with environment variables
    if env_settings.openai_api_key:
        config.openai.api_key = env_settings.openai_api_key

    if env_settings.openai_model:
        config.openai.model = env_settings.openai_model

    if env_settings.hf_token:
        config.huggingface.token = env_settings.hf_token

    if env_settings.hf_home:
        config.huggingface.cache_dir = env_settings.hf_home

    return config


def get_default_config() -> LoCLIConfig:
    """Get default configuration."""
    return LoCLIConfig()


# VRAM estimation lookup table (in GB)
MODEL_VRAM_ESTIMATES = {
    # Format: (model_size_billions, method): vram_gb
    (7, "qlora"): 6,
    (7, "lora"): 14,
    (8, "qlora"): 8,
    (8, "lora"): 16,
    (13, "qlora"): 10,
    (13, "lora"): 26,
    (14, "qlora"): 12,
    (14, "lora"): 28,
    (70, "qlora"): 24,
    (70, "lora"): 140,
    (72, "qlora"): 26,
    (72, "lora"): 144,
}


def estimate_vram(model_size_b: float, method: Literal["lora", "qlora"]) -> float:
    """Estimate VRAM requirements for a model.

    Args:
        model_size_b: Model size in billions of parameters
        method: Training method (lora or qlora)

    Returns:
        Estimated VRAM in GB
    """
    # Find the closest model size in our lookup table
    sizes = sorted(set(k[0] for k in MODEL_VRAM_ESTIMATES.keys()))
    closest_size = min(sizes, key=lambda x: abs(x - model_size_b))

    key = (closest_size, method)
    if key in MODEL_VRAM_ESTIMATES:
        # Scale linearly based on actual model size
        base_vram = MODEL_VRAM_ESTIMATES[key]
        scale_factor = model_size_b / closest_size
        return base_vram * scale_factor

    # Fallback: rough estimation
    if method == "qlora":
        return model_size_b * 0.8  # ~0.8GB per billion params for QLoRA
    else:
        return model_size_b * 2.0  # ~2GB per billion params for LoRA
