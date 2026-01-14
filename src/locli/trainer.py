"""Training pipeline for LoCLI using PEFT and TRL."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import SFTTrainer

from locli.config import (
    EarlyStoppingConfig,
    HardwareConfig,
    LoCLIConfig,
    LoRAConfig,
    TrainingConfig,
    load_config,
)
from locli.utils import get_hf_token, print_error, print_info, print_success, set_seed

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

console = Console()


@dataclass
class TrainingResult:
    """Result of a training run."""

    output_dir: Path
    final_loss: float | None
    total_steps: int
    epochs_completed: float
    best_checkpoint: Path | None
    training_time_seconds: float


def get_chat_template(model_id: str) -> str | None:
    """Get the appropriate chat template for a model."""
    model_lower = model_id.lower()

    # Llama 3 template
    if "llama-3" in model_lower or "llama3" in model_lower:
        return None  # Use model's built-in template

    # Mistral template
    if "mistral" in model_lower:
        return None  # Use model's built-in template

    # Qwen template
    if "qwen" in model_lower:
        return None  # Use model's built-in template

    # Phi template
    if "phi" in model_lower:
        return None  # Use model's built-in template

    return None


def format_chat_sample(sample: dict, tokenizer: "PreTrainedTokenizer") -> str:
    """Format a chat sample using the tokenizer's chat template."""
    if "messages" in sample:
        return tokenizer.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=False
        )

    # Convert instruction format to messages
    messages = []

    if "system" in sample:
        messages.append({"role": "system", "content": sample["system"]})

    if "instruction" in sample:
        user_content = sample["instruction"]
        if "input" in sample and sample["input"]:
            user_content += f"\n\n{sample['input']}"
        messages.append({"role": "user", "content": user_content})

    if "output" in sample:
        messages.append({"role": "assistant", "content": sample["output"]})
    elif "response" in sample:
        messages.append({"role": "assistant", "content": sample["response"]})

    if messages:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    # Fallback: just concatenate text
    if "prompt" in sample and "completion" in sample:
        return f"{sample['prompt']}{sample['completion']}"

    if "text" in sample:
        return sample["text"]

    return str(sample)


def load_training_dataset(
    dataset_path: Path, tokenizer: "PreTrainedTokenizer"
) -> Dataset:
    """Load and preprocess the training dataset.

    Args:
        dataset_path: Path to the JSONL dataset
        tokenizer: Tokenizer for formatting

    Returns:
        HuggingFace Dataset
    """
    # Load the JSONL file
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    # Format samples
    def format_sample(sample: dict) -> dict:
        text = format_chat_sample(sample, tokenizer)
        return {"text": text}

    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    return dataset


def create_quantization_config(method: Literal["lora", "qlora"]) -> BitsAndBytesConfig | None:
    """Create quantization configuration for QLoRA."""
    if method == "qlora":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return None


def create_lora_config(config: LoRAConfig) -> LoraConfig:
    """Create PEFT LoRA configuration."""
    return LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias=config.bias,
        task_type=config.task_type,
    )


def load_model_and_tokenizer(
    model_id: str,
    method: Literal["lora", "qlora"],
    hardware_config: HardwareConfig,
) -> tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    """Load the base model and tokenizer.

    Args:
        model_id: HuggingFace model ID
        method: Training method (lora or qlora)
        hardware_config: Hardware configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    token = get_hf_token()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=token,
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create quantization config
    quantization_config = create_quantization_config(method)

    # Determine dtype
    if hardware_config.bf16:
        torch_dtype = torch.bfloat16
    elif hardware_config.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Check if Flash Attention 2 is available
    attn_implementation = None
    if torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            attn_implementation = "flash_attention_2"
        except ImportError:
            pass  # Flash Attention not installed, use default

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        token=token,
        attn_implementation=attn_implementation,
    )

    # Enable gradient checkpointing
    if hardware_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Prepare model for k-bit training if using QLoRA
    if method == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=hardware_config.gradient_checkpointing,
        )

    return model, tokenizer


def create_training_arguments(
    output_dir: Path,
    training_config: TrainingConfig,
    hardware_config: HardwareConfig,
    early_stopping: EarlyStoppingConfig,
) -> TrainingArguments:
    """Create HuggingFace TrainingArguments."""
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_ratio=training_config.warmup_ratio,
        weight_decay=training_config.weight_decay,
        max_grad_norm=training_config.max_grad_norm,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        save_total_limit=3,
        bf16=hardware_config.bf16,
        fp16=hardware_config.fp16,
        gradient_checkpointing=hardware_config.gradient_checkpointing,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to=[],  # Disable wandb by default
        load_best_model_at_end=early_stopping.enabled,
        metric_for_best_model="loss" if early_stopping.enabled else None,
        greater_is_better=False if early_stopping.enabled else None,
        evaluation_strategy="steps" if early_stopping.enabled else "no",
        eval_steps=training_config.save_steps if early_stopping.enabled else None,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )


class TrainingProgressCallback:
    """Callback for displaying training progress."""

    def __init__(self):
        self.progress = None
        self.task_id = None
        self.current_step = 0
        self.total_steps = 0
        self.current_loss = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.total_steps = state.max_steps
        console.print(f"\n[bold green]Starting training for {self.total_steps} steps...[/bold green]\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging."""
        if logs:
            loss = logs.get("loss")
            if loss is not None:
                self.current_loss = loss
                lr = logs.get("learning_rate", 0)
                step = state.global_step
                epoch = state.epoch or 0

                console.print(
                    f"Step {step}/{self.total_steps} | "
                    f"Epoch {epoch:.2f} | "
                    f"Loss: {loss:.4f} | "
                    f"LR: {lr:.2e}"
                )

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        console.print(f"\n[bold green]Training complete! Final loss: {self.current_loss:.4f}[/bold green]")


def train(
    dataset_path: Path,
    base_model: str,
    output_dir: Path,
    method: Literal["lora", "qlora"] = "qlora",
    config: LoCLIConfig | None = None,
    resume_from: Path | None = None,
    seed: int = 42,
) -> TrainingResult:
    """Run the fine-tuning training pipeline.

    Args:
        dataset_path: Path to the JSONL training dataset
        base_model: HuggingFace model ID
        output_dir: Directory to save the trained model
        method: Training method (lora or qlora)
        config: LoCLI configuration (uses defaults if None)
        resume_from: Path to checkpoint to resume from
        seed: Random seed for reproducibility

    Returns:
        TrainingResult with training statistics
    """
    import time

    start_time = time.time()

    # Set seed for reproducibility
    set_seed(seed)

    # Load configuration
    if config is None:
        config = load_config()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_info(f"Loading model: {base_model}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading model and tokenizer...", total=None)

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            base_model, method, config.hardware
        )

        progress.update(task, description="Model loaded!")

    # Create LoRA config and apply PEFT
    lora_config = create_lora_config(config.lora)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print_info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    print_info(f"Loading dataset: {dataset_path}")

    # Load dataset
    dataset = load_training_dataset(dataset_path, tokenizer)

    # Split for evaluation if early stopping enabled
    if config.early_stopping.enabled:
        split = dataset.train_test_split(test_size=0.1, seed=seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    print_info(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        print_info(f"Evaluation samples: {len(eval_dataset)}")

    # Create training arguments
    training_args = create_training_arguments(
        output_dir, config.training, config.hardware, config.early_stopping
    )

    # Create callbacks
    callbacks = []
    if config.early_stopping.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping.patience,
                early_stopping_threshold=config.early_stopping.min_delta,
            )
        )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=config.training.max_seq_length,
        dataset_text_field="text",
        callbacks=callbacks,
    )

    # Resume from checkpoint if specified
    if resume_from:
        print_info(f"Resuming from checkpoint: {resume_from}")

    # Train
    console.print()
    train_result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)

    # Save the final model
    final_output = output_dir / "final"
    trainer.save_model(str(final_output))
    tokenizer.save_pretrained(str(final_output))

    # Save training info
    training_info = {
        "base_model": base_model,
        "method": method,
        "lora_config": {
            "r": config.lora.r,
            "lora_alpha": config.lora.lora_alpha,
            "lora_dropout": config.lora.lora_dropout,
            "target_modules": config.lora.target_modules,
        },
        "training_config": {
            "learning_rate": config.training.learning_rate,
            "num_epochs": config.training.num_epochs,
            "batch_size": config.training.batch_size,
            "max_seq_length": config.training.max_seq_length,
        },
        "total_steps": trainer.state.global_step,
        "final_loss": train_result.training_loss,
    }

    with open(output_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    end_time = time.time()
    training_time = end_time - start_time

    print_success(f"Model saved to: {final_output}")

    # Find best checkpoint
    best_checkpoint = None
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if checkpoints:
        best_checkpoint = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))

    return TrainingResult(
        output_dir=output_dir,
        final_loss=train_result.training_loss,
        total_steps=trainer.state.global_step,
        epochs_completed=trainer.state.epoch or config.training.num_epochs,
        best_checkpoint=best_checkpoint,
        training_time_seconds=training_time,
    )


def get_available_checkpoints(output_dir: Path) -> list[Path]:
    """Get list of available checkpoints in output directory.

    Args:
        output_dir: Training output directory

    Returns:
        List of checkpoint paths, sorted by step number
    """
    checkpoints = list(Path(output_dir).glob("checkpoint-*"))
    return sorted(checkpoints, key=lambda p: int(p.name.split("-")[1]))


def display_training_result(result: TrainingResult) -> None:
    """Display training result summary."""
    from rich.table import Table

    table = Table(title="Training Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Output Directory", str(result.output_dir))
    table.add_row("Total Steps", f"{result.total_steps:,}")
    table.add_row("Epochs Completed", f"{result.epochs_completed:.2f}")

    if result.final_loss:
        table.add_row("Final Loss", f"{result.final_loss:.4f}")

    # Format training time
    minutes = result.training_time_seconds / 60
    if minutes < 60:
        time_str = f"{minutes:.1f} minutes"
    else:
        hours = minutes / 60
        time_str = f"{hours:.1f} hours"
    table.add_row("Training Time", time_str)

    if result.best_checkpoint:
        table.add_row("Best Checkpoint", result.best_checkpoint.name)

    console.print(table)
