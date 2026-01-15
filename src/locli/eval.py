"""Evaluation and testing module for LoCLI."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

console = Console()


@dataclass
class TrainingMetrics:
    """Training metrics collected during training."""

    steps: list[int]
    losses: list[float]
    learning_rates: list[float]
    epochs: list[float]
    eval_losses: list[float] | None = None
    eval_steps: list[int] | None = None


@dataclass
class EvalResult:
    """Evaluation results."""

    perplexity: float
    avg_loss: float
    num_samples: int


def save_training_metrics(metrics: TrainingMetrics, output_dir: Path) -> Path:
    """Save training metrics to JSON file.

    Args:
        metrics: Training metrics
        output_dir: Output directory

    Returns:
        Path to saved metrics file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "training_metrics.json"

    data = {
        "steps": metrics.steps,
        "losses": metrics.losses,
        "learning_rates": metrics.learning_rates,
        "epochs": metrics.epochs,
    }

    if metrics.eval_losses:
        data["eval_losses"] = metrics.eval_losses
        data["eval_steps"] = metrics.eval_steps

    with open(metrics_path, "w") as f:
        json.dump(data, f, indent=2)

    return metrics_path


def load_training_metrics(output_dir: Path) -> TrainingMetrics | None:
    """Load training metrics from JSON file.

    Args:
        output_dir: Output directory containing metrics

    Returns:
        TrainingMetrics or None if not found
    """
    metrics_path = Path(output_dir) / "training_metrics.json"

    if not metrics_path.exists():
        return None

    with open(metrics_path) as f:
        data = json.load(f)

    return TrainingMetrics(
        steps=data["steps"],
        losses=data["losses"],
        learning_rates=data["learning_rates"],
        epochs=data["epochs"],
        eval_losses=data.get("eval_losses"),
        eval_steps=data.get("eval_steps"),
    )


def generate_training_charts(metrics: TrainingMetrics, output_dir: Path) -> list[Path]:
    """Generate training charts from metrics.

    Args:
        metrics: Training metrics
        output_dir: Output directory for charts

    Returns:
        List of paths to generated chart images
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not installed. Install with: pip install matplotlib[/yellow]")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts = []

    # Set style
    plt.style.use('dark_background')

    # 1. Training Loss Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics.steps, metrics.losses, 'c-', linewidth=1.5, label='Training Loss')

    if metrics.eval_losses and metrics.eval_steps:
        ax.plot(metrics.eval_steps, metrics.eval_losses, 'r-', linewidth=2, label='Eval Loss', marker='o')

    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    loss_path = output_dir / "loss_chart.png"
    plt.savefig(loss_path, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
    plt.close()
    charts.append(loss_path)

    # 2. Learning Rate Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics.steps, metrics.learning_rates, 'm-', linewidth=1.5)
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    lr_path = output_dir / "learning_rate_chart.png"
    plt.savefig(lr_path, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
    plt.close()
    charts.append(lr_path)

    # 3. Loss by Epoch Chart
    if len(set(metrics.epochs)) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by epoch
        epoch_losses = {}
        for epoch, loss in zip(metrics.epochs, metrics.losses):
            epoch_int = int(epoch)
            if epoch_int not in epoch_losses:
                epoch_losses[epoch_int] = []
            epoch_losses[epoch_int].append(loss)

        # Average loss per epoch
        epochs = sorted(epoch_losses.keys())
        avg_losses = [sum(epoch_losses[e]) / len(epoch_losses[e]) for e in epochs]

        ax.bar(epochs, avg_losses, color='cyan', alpha=0.7, edgecolor='white')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Average Loss', fontsize=12)
        ax.set_title('Average Loss by Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        epoch_path = output_dir / "epoch_loss_chart.png"
        plt.savefig(epoch_path, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
        plt.close()
        charts.append(epoch_path)

    return charts


def display_training_summary(metrics: TrainingMetrics) -> None:
    """Display a summary of training metrics."""
    table = Table(title="Training Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Total Steps", f"{len(metrics.steps):,}")
    table.add_row("Final Loss", f"{metrics.losses[-1]:.4f}")
    table.add_row("Best Loss", f"{min(metrics.losses):.4f}")
    table.add_row("Initial Loss", f"{metrics.losses[0]:.4f}")
    table.add_row("Loss Reduction", f"{((metrics.losses[0] - metrics.losses[-1]) / metrics.losses[0] * 100):.1f}%")

    if metrics.eval_losses:
        table.add_row("Final Eval Loss", f"{metrics.eval_losses[-1]:.4f}")
        table.add_row("Best Eval Loss", f"{min(metrics.eval_losses):.4f}")

    console.print(table)


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from loss."""
    return math.exp(loss)


def load_model_for_inference(
    model_path: Path,
    base_model: str | None = None,
) -> tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    """Load a trained model for inference.

    Args:
        model_path: Path to the trained model
        base_model: Base model ID (optional, will try to detect)

    Returns:
        Tuple of (model, tokenizer)
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_path)

    # Check if it's a LoRA adapter or full model
    adapter_config = model_path / "adapter_config.json"

    if adapter_config.exists():
        # Load adapter config to get base model
        with open(adapter_config) as f:
            config = json.load(f)
            base_model = base_model or config.get("base_model_name_or_path")

        if not base_model:
            raise ValueError("Could not determine base model. Please specify --base-model.")

        console.print(f"[dim]Loading base model: {base_model}[/dim]")

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load adapter
        model = PeftModel.from_pretrained(model, str(model_path))

        # Load tokenizer from adapter path
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    else:
        # Full model
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def generate_response(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a response from the model.

    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling

    Returns:
        Generated text
    """
    # Format as chat if the tokenizer supports it
    try:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from response
    if response.startswith(formatted.replace(tokenizer.bos_token or "", "")):
        response = response[len(formatted):]

    return response.strip()


def interactive_test(model_path: Path, base_model: str | None = None) -> None:
    """Run interactive testing session with the model.

    Args:
        model_path: Path to the trained model
        base_model: Base model ID (optional)
    """
    from rich.prompt import Prompt

    console.print()
    console.print(Panel.fit("[bold]Interactive Model Testing[/bold]", border_style="green"))
    console.print()
    console.print("[dim]Loading model...[/dim]")

    model, tokenizer = load_model_for_inference(model_path, base_model)

    console.print("[green]Model loaded! Type 'exit' or 'quit' to end.[/green]")
    console.print("[dim]Tip: Use 'clear' to clear the screen[/dim]")
    console.print()

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")

            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "clear":
                console.clear()
                continue

            if not user_input.strip():
                continue

            with console.status("[dim]Generating response...[/dim]"):
                response = generate_response(model, tokenizer, user_input)

            console.print()
            console.print(f"[bold green]Assistant[/bold green]: {response}")
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break


def run_eval_samples(
    model_path: Path,
    samples: list[str],
    base_model: str | None = None,
) -> None:
    """Run evaluation on sample prompts.

    Args:
        model_path: Path to the trained model
        samples: List of sample prompts to test
        base_model: Base model ID (optional)
    """
    console.print()
    console.print(Panel.fit("[bold]Model Evaluation[/bold]", border_style="blue"))
    console.print()
    console.print("[dim]Loading model...[/dim]")

    model, tokenizer = load_model_for_inference(model_path, base_model)

    console.print(f"[green]Running {len(samples)} test samples...[/green]")
    console.print()

    for i, prompt in enumerate(samples, 1):
        console.print(f"[bold cyan]Sample {i}/{len(samples)}[/bold cyan]")
        console.print(f"[dim]Prompt:[/dim] {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        with console.status("[dim]Generating...[/dim]"):
            response = generate_response(model, tokenizer, prompt, max_new_tokens=150)

        console.print(f"[green]Response:[/green] {response}")
        console.print()
