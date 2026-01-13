"""CLI entry point for LoCLI using Typer."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from locli import __version__
from locli.analyzer import (
    analyze_dataset,
    display_stats,
    display_suggestions,
    get_suggestions,
    validate_jsonl,
)
from locli.config import load_config
from locli.exporter import (
    GGUF_QUANTIZATIONS,
    display_export_result,
    display_gguf_quantizations,
    export_model,
)
from locli.models import (
    display_model_families,
    display_model_info,
    display_model_list,
    get_model_info,
    recommend_method,
    search_models,
    validate_model_id,
)
from locli.trainer import display_training_result, get_available_checkpoints, train
from locli.utils import (
    check_cuda_available,
    display_system_info,
    get_available_vram,
    get_system_info,
    get_total_vram,
    print_error,
    print_info,
    print_success,
    print_warning,
    validate_dataset_path,
    validate_output_dir,
)

console = Console()

# Create the main app
app = typer.Typer(
    name="locli",
    help="Fine-tune LLMs locally with AI-optimized defaults",
    add_completion=False,
    no_args_is_help=True,
)

# Create subcommand groups
models_app = typer.Typer(help="Browse and search HuggingFace models")
app.add_typer(models_app, name="models")


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"LoCLI version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """LoCLI - Fine-tune LLMs locally with AI-optimized defaults."""
    pass


@app.command()
def info() -> None:
    """Show system information (GPU, VRAM, CUDA)."""
    console.print()
    console.print(Panel.fit("[bold]LoCLI System Information[/bold]", border_style="blue"))
    console.print()

    system_info = get_system_info()
    display_system_info(system_info)

    if not system_info.cuda_available:
        console.print()
        print_warning("CUDA is not available. Training will be very slow on CPU.")
        print_info("Make sure you have an NVIDIA GPU with CUDA installed.")


@app.command()
def train_cmd(
    dataset: Annotated[
        Path,
        typer.Option(
            "--dataset",
            "-d",
            help="Path to JSONL dataset",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    base_model: Annotated[
        str,
        typer.Option(
            "--base-model",
            "-m",
            help="HuggingFace model ID (e.g., meta-llama/Llama-3.2-8B-Instruct)",
        ),
    ] = None,
    method: Annotated[
        str,
        typer.Option(
            "--method",
            help="Training method: lora or qlora",
        ),
    ] = "qlora",
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for trained model",
        ),
    ] = Path("./output"),
    epochs: Annotated[
        int,
        typer.Option("--epochs", help="Number of training epochs"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Batch size per device"),
    ] = None,
    lr: Annotated[
        float,
        typer.Option("--lr", help="Learning rate"),
    ] = None,
    r: Annotated[
        int,
        typer.Option("--r", help="LoRA rank"),
    ] = None,
    resume: Annotated[
        Path,
        typer.Option("--resume", help="Resume from checkpoint"),
    ] = None,
    interactive: Annotated[
        bool,
        typer.Option("--interactive", "-i", help="Interactive mode"),
    ] = False,
) -> None:
    """Start fine-tuning a model."""
    console.print()
    console.print(Panel.fit("[bold]LoCLI Training[/bold]", border_style="blue"))
    console.print()

    # Check CUDA availability
    if not check_cuda_available():
        print_warning("CUDA is not available. Training will be very slow on CPU.")
        if not Confirm.ask("Continue anyway?", default=False):
            raise typer.Exit(1)

    # Interactive mode
    if interactive or (dataset is None and base_model is None):
        dataset, base_model, method, output = interactive_training_setup()

    # Validate inputs
    if dataset is None:
        print_error("Dataset path is required. Use --dataset or -d")
        raise typer.Exit(1)

    if base_model is None:
        print_error("Base model is required. Use --base-model or -m")
        raise typer.Exit(1)

    # Validate dataset
    try:
        dataset_path = validate_dataset_path(dataset)
    except (FileNotFoundError, ValueError) as e:
        print_error(str(e))
        raise typer.Exit(1)

    # Validate method
    if method not in ["lora", "qlora"]:
        print_error(f"Invalid method: {method}. Use 'lora' or 'qlora'")
        raise typer.Exit(1)

    # Check VRAM and recommend method
    available_vram = get_total_vram()
    if available_vram > 0:
        recommended = recommend_method(base_model, available_vram)
        if recommended is None:
            print_warning(
                f"Model may be too large for available VRAM ({available_vram:.0f} GB). "
                "Consider using a smaller model or QLoRA."
            )
        elif recommended != method:
            print_info(
                f"Based on available VRAM ({available_vram:.0f} GB), "
                f"'{recommended}' is recommended over '{method}'."
            )

    # Validate output directory
    try:
        output_dir = validate_output_dir(output)
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    # Load config and apply overrides
    config = load_config()

    if epochs is not None:
        config.training.num_epochs = epochs
    if batch_size is not None:
        config.training.batch_size = batch_size
    if lr is not None:
        config.training.learning_rate = lr
    if r is not None:
        config.lora.r = r

    # Display configuration
    console.print()
    print_info(f"Dataset: {dataset_path}")
    print_info(f"Base model: {base_model}")
    print_info(f"Method: {method}")
    print_info(f"Output: {output_dir}")
    print_info(f"Epochs: {config.training.num_epochs}")
    print_info(f"Batch size: {config.training.batch_size}")
    print_info(f"Learning rate: {config.training.learning_rate}")
    print_info(f"LoRA rank: {config.lora.r}")
    console.print()

    if not Confirm.ask("Start training?", default=True):
        raise typer.Exit(0)

    # Run training
    try:
        result = train(
            dataset_path=dataset_path,
            base_model=base_model,
            output_dir=output_dir,
            method=method,
            config=config,
            resume_from=resume,
        )
        console.print()
        display_training_result(result)

    except Exception as e:
        print_error(f"Training failed: {e}")
        raise typer.Exit(1)


def interactive_training_setup() -> tuple[Path, str, str, Path]:
    """Run interactive training setup wizard."""
    console.print("[bold]Interactive Training Setup[/bold]")
    console.print()

    # Get dataset path
    dataset_str = Prompt.ask("Dataset path (JSONL file)")
    dataset_path = Path(dataset_str)

    if not dataset_path.exists():
        print_error(f"Dataset not found: {dataset_path}")
        raise typer.Exit(1)

    # Validate and analyze dataset
    valid, msg = validate_jsonl(dataset_path)
    if not valid:
        print_error(f"Invalid dataset: {msg}")
        raise typer.Exit(1)

    print_success(msg)

    # Analyze dataset
    console.print()
    stats = analyze_dataset(dataset_path)
    display_stats(stats)

    # Get model
    console.print()
    console.print("[bold]Model Selection[/bold]")
    console.print("Supported families: Llama, Mistral, Qwen, Phi")
    console.print()

    base_model = Prompt.ask(
        "Base model (HuggingFace ID)",
        default="meta-llama/Llama-3.2-8B-Instruct",
    )

    # Validate model exists
    console.print("Validating model...")
    if not validate_model_id(base_model):
        print_warning(f"Could not validate model: {base_model}")
        if not Confirm.ask("Continue anyway?", default=False):
            raise typer.Exit(1)

    # Recommend method based on VRAM
    available_vram = get_total_vram()
    if available_vram > 0:
        recommended = recommend_method(base_model, available_vram)
        if recommended:
            method = recommended
            print_info(f"Recommended method for {available_vram:.0f}GB VRAM: {method}")
        else:
            method = "qlora"
            print_warning("Model may be too large. Using QLoRA.")
    else:
        method = "qlora"

    method = Prompt.ask("Training method", choices=["lora", "qlora"], default=method)

    # Get suggestions
    console.print()
    available_vram = get_available_vram() or 8  # Default to 8GB if unknown
    suggestions = get_suggestions(stats, available_vram)
    is_ai = hasattr(suggestions, "reasoning") and "AI" in suggestions.reasoning
    display_suggestions(suggestions, is_ai)

    # Output directory
    console.print()
    output_str = Prompt.ask("Output directory", default="./output")
    output_path = Path(output_str)

    return dataset_path, base_model, method, output_path


@app.command()
def analyze(
    dataset: Annotated[
        Path,
        typer.Argument(help="Path to JSONL dataset"),
    ],
    suggest: Annotated[
        bool,
        typer.Option("--suggest", "-s", help="Get training parameter suggestions"),
    ] = False,
) -> None:
    """Analyze a dataset and optionally get training suggestions."""
    console.print()
    console.print(Panel.fit("[bold]Dataset Analysis[/bold]", border_style="blue"))
    console.print()

    # Validate dataset
    try:
        dataset_path = validate_dataset_path(dataset)
    except (FileNotFoundError, ValueError) as e:
        print_error(str(e))
        raise typer.Exit(1)

    # Validate format
    valid, msg = validate_jsonl(dataset_path)
    if not valid:
        print_error(msg)
        raise typer.Exit(1)

    print_success(msg)
    console.print()

    # Analyze
    try:
        stats = analyze_dataset(dataset_path)
        display_stats(stats)
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        raise typer.Exit(1)

    # Get suggestions if requested
    if suggest:
        console.print()
        available_vram = get_available_vram() or 8
        suggestions = get_suggestions(stats, available_vram, use_ai=True)
        is_ai = "AI" in suggestions.reasoning if suggestions.reasoning else False
        display_suggestions(suggestions, is_ai)


@app.command()
def export(
    model_path: Annotated[
        Path,
        typer.Argument(help="Path to trained model directory"),
    ],
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Export format: lora, merged, or gguf"),
    ] = "lora",
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path"),
    ] = None,
    quantization: Annotated[
        str,
        typer.Option("--quantization", "-q", help="GGUF quantization type"),
    ] = "q4_k_m",
    base_model: Annotated[
        str,
        typer.Option("--base-model", help="Base model ID (for gguf export)"),
    ] = None,
    list_quantizations: Annotated[
        bool,
        typer.Option("--list-quantizations", help="List available GGUF quantizations"),
    ] = False,
) -> None:
    """Export trained model to different formats."""
    if list_quantizations:
        display_gguf_quantizations()
        raise typer.Exit(0)

    console.print()
    console.print(Panel.fit("[bold]Model Export[/bold]", border_style="blue"))
    console.print()

    # Validate model path
    if not model_path.exists():
        print_error(f"Model path not found: {model_path}")
        raise typer.Exit(1)

    # Validate format
    if format not in ["lora", "merged", "gguf"]:
        print_error(f"Invalid format: {format}. Use 'lora', 'merged', or 'gguf'")
        raise typer.Exit(1)

    # Validate quantization for GGUF
    if format == "gguf" and quantization not in GGUF_QUANTIZATIONS:
        print_error(f"Invalid quantization: {quantization}")
        display_gguf_quantizations()
        raise typer.Exit(1)

    # Set default output path
    if output is None:
        if format == "gguf":
            output = model_path.parent / f"{model_path.stem}.gguf"
        else:
            output = model_path.parent / f"{model_path.stem}-{format}"

    print_info(f"Model: {model_path}")
    print_info(f"Format: {format}")
    print_info(f"Output: {output}")
    if format == "gguf":
        print_info(f"Quantization: {quantization}")
    console.print()

    # Export
    try:
        result = export_model(
            model_path=model_path,
            output_path=output,
            format=format,
            quantization=quantization if format == "gguf" else None,
            base_model=base_model,
        )
        console.print()
        display_export_result(result)

    except Exception as e:
        print_error(f"Export failed: {e}")
        raise typer.Exit(1)


# Models subcommands
@models_app.command("list")
def models_list() -> None:
    """List supported model families."""
    console.print()
    display_model_families()


@models_app.command("search")
def models_search(
    query: Annotated[
        str,
        typer.Argument(help="Search query"),
    ] = "",
    family: Annotated[
        str,
        typer.Option("--family", "-f", help="Filter by family (llama, mistral, qwen, phi)"),
    ] = None,
    all_models: Annotated[
        bool,
        typer.Option("--all", "-a", help="Include base models (not just instruct)"),
    ] = False,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum results"),
    ] = 20,
) -> None:
    """Search HuggingFace models."""
    console.print()
    console.print(f"[bold]Searching for: {query or 'all models'}[/bold]")
    console.print()

    models = search_models(
        query=query,
        family=family,
        instruct_only=not all_models,
        limit=limit,
    )

    display_model_list(models)


@models_app.command("info")
def models_info(
    model_id: Annotated[
        str,
        typer.Argument(help="HuggingFace model ID"),
    ],
) -> None:
    """Show detailed model information."""
    model = get_model_info(model_id)

    if model is None:
        print_error(f"Model not found: {model_id}")
        raise typer.Exit(1)

    display_model_info(model)

    # Show method recommendation
    available_vram = get_total_vram()
    if available_vram > 0:
        recommended = recommend_method(model_id, available_vram)
        console.print()
        if recommended:
            print_info(f"Recommended training method: {recommended} (for {available_vram:.0f}GB VRAM)")
        else:
            print_warning("Model may be too large for available VRAM")


# Register the train command with the right name
app.command(name="train")(train_cmd)


if __name__ == "__main__":
    app()
