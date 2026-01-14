"""CLI entry point for LoCLI using Typer."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

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
from locli.trainer import display_training_result, train
from locli.utils import (
    check_cuda_available,
    display_system_info,
    fix_pytorch_cuda,
    get_available_vram,
    get_hf_token,
    get_system_info,
    get_total_vram,
    has_nvidia_gpu,
    print_error,
    print_info,
    print_success,
    print_warning,
    validate_dataset_path,
)

console = Console()

LOGO = """
[bold cyan]
  ██╗      ██████╗  ██████╗██╗     ██╗
  ██║     ██╔═══██╗██╔════╝██║     ██║
  ██║     ██║   ██║██║     ██║     ██║
  ██║     ██║   ██║██║     ██║     ██║
  ███████╗╚██████╔╝╚██████╗███████╗██║
  ╚══════╝ ╚═════╝  ╚═════╝╚══════╝╚═╝
[/bold cyan]
[dim]  Fine-tune LLMs locally with AI-optimized defaults[/dim]
[dim italic]  by t21.dev[/dim italic]
"""


def show_logo() -> None:
    """Display the LoCLI logo."""
    console.print(LOGO)


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
    show_logo()
    console.print(Panel.fit("[bold]System Information[/bold]", border_style="blue"))
    console.print()

    system_info = get_system_info()
    display_system_info(system_info)

    if not system_info.cuda_available:
        console.print()
        print_warning("CUDA is not available. Training will be very slow on CPU.")
        print_info("Make sure you have an NVIDIA GPU with CUDA installed.")


@app.command(name="train")
def train_cmd() -> None:
    """Start fine-tuning with interactive setup."""
    show_logo()
    console.print(Panel.fit("[bold]Training Wizard[/bold]", border_style="blue"))
    console.print()

    # Check CUDA availability
    if not check_cuda_available():
        if has_nvidia_gpu():
            # GPU exists but PyTorch doesn't have CUDA - offer to fix
            if fix_pytorch_cuda():
                raise typer.Exit(0)  # User needs to restart
            # User declined fix, ask if they want to continue on CPU
            if not Confirm.ask("Continue with CPU training? (very slow)", default=False):
                raise typer.Exit(1)
        else:
            print_warning("No NVIDIA GPU detected. Training will be very slow on CPU.")
            if not Confirm.ask("Continue anyway?", default=False):
                raise typer.Exit(1)

    # Step 1: Get dataset path
    console.print("[bold]Step 1: Dataset[/bold]")
    console.print()

    while True:
        dataset_str = Prompt.ask("Dataset path (JSONL file)")
        dataset_path = Path(dataset_str)

        if not dataset_path.exists():
            print_error(f"File not found: {dataset_path}")
            continue

        # Validate format
        valid, msg = validate_jsonl(dataset_path)
        if not valid:
            print_error(f"Invalid dataset: {msg}")
            continue

        print_success(msg)
        break

    # Analyze dataset
    console.print()
    stats = analyze_dataset(dataset_path)
    display_stats(stats)

    # Step 2: Model selection
    console.print()
    console.print("[bold]Step 2: Model Selection[/bold]")
    console.print("Supported families: Llama, Mistral, Qwen, Phi")
    console.print()

    while True:
        base_model = Prompt.ask(
            "Base model (HuggingFace ID)",
            default="meta-llama/Llama-3.2-3B-Instruct",
        )

        console.print("Validating model...")
        if validate_model_id(base_model):
            print_success(f"Model found: {base_model}")
            break
        else:
            print_warning(f"Could not validate model: {base_model}")
            if Confirm.ask("Use anyway?", default=False):
                break

    # Show model info
    model_info = get_model_info(base_model)
    if model_info:
        display_model_info(model_info)

    # Check HF token for gated models (Llama, etc.)
    if "llama" in base_model.lower() or "mistral" in base_model.lower():
        hf_token = get_hf_token()
        if not hf_token:
            console.print()
            print_warning("This model requires HuggingFace authentication.")
            print_info("1. Go to https://huggingface.co/settings/tokens and create a token")
            print_info("2. Request access to the model at https://huggingface.co/" + base_model)
            print_info("3. Add HF_TOKEN=your_token to your .env file")
            console.print()
            if not Confirm.ask("Do you have access and HF_TOKEN configured?", default=False):
                raise typer.Exit(1)
        else:
            print_success("HuggingFace token found")

    # Step 3: Training method
    console.print()
    console.print("[bold]Step 3: Training Method[/bold]")
    console.print()

    available_vram = get_total_vram()
    recommended_method = "qlora"

    if available_vram > 0:
        recommended = recommend_method(base_model, available_vram)
        if recommended:
            recommended_method = recommended
            print_info(f"Based on your VRAM ({available_vram:.0f}GB), recommended: {recommended}")
        else:
            print_warning("Model may be too large for your VRAM. Using QLoRA.")

    method = Prompt.ask(
        "Training method",
        choices=["lora", "qlora"],
        default=recommended_method,
    )

    # Step 4: Training parameters
    console.print()
    console.print("[bold]Step 4: Training Parameters[/bold]")
    console.print()

    # Get AI/static suggestions with loading indicator
    vram_for_suggestions = get_available_vram() or 8

    with console.status("[bold cyan]Analyzing dataset and generating optimal parameters...[/bold cyan]", spinner="dots"):
        suggestions = get_suggestions(stats, vram_for_suggestions, use_ai=True)

    is_ai = "AI" in suggestions.reasoning if suggestions.reasoning else False
    display_suggestions(suggestions, is_ai)

    console.print()
    if Confirm.ask("Use suggested parameters?", default=True):
        epochs = suggestions.num_epochs
        batch_size = suggestions.batch_size
        learning_rate = suggestions.learning_rate
        lora_rank = suggestions.r
        max_seq_length = suggestions.max_seq_length
    else:
        epochs = int(Prompt.ask("Number of epochs", default=str(suggestions.num_epochs)))
        batch_size = int(Prompt.ask("Batch size", default=str(suggestions.batch_size)))
        learning_rate = float(Prompt.ask("Learning rate", default=str(suggestions.learning_rate)))
        lora_rank = int(Prompt.ask("LoRA rank", default=str(suggestions.r)))
        max_seq_length = int(Prompt.ask("Max sequence length", default=str(suggestions.max_seq_length)))

    # Step 5: Output directory
    console.print()
    console.print("[bold]Step 5: Output[/bold]")
    console.print()

    output_str = Prompt.ask("Output directory", default="./output")
    output_dir = Path(output_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary and confirmation
    console.print()
    console.print(Panel("[bold]Training Configuration[/bold]", border_style="green"))
    console.print()
    console.print(f"  Dataset:        {dataset_path}")
    console.print(f"  Samples:        {stats.total_samples}")
    console.print(f"  Base model:     {base_model}")
    console.print(f"  Method:         {method}")
    console.print(f"  Epochs:         {epochs}")
    console.print(f"  Batch size:     {batch_size}")
    console.print(f"  Learning rate:  {learning_rate}")
    console.print(f"  LoRA rank:      {lora_rank}")
    console.print(f"  Max seq length: {max_seq_length}")
    console.print(f"  Output:         {output_dir}")
    console.print()

    if not Confirm.ask("Start training?", default=True):
        console.print("Training cancelled.")
        raise typer.Exit(0)

    # Load config and apply settings
    config = load_config()
    config.training.num_epochs = epochs
    config.training.batch_size = batch_size
    config.training.learning_rate = learning_rate
    config.training.max_seq_length = max_seq_length
    config.lora.r = lora_rank

    # Run training
    try:
        result = train(
            dataset_path=dataset_path,
            base_model=base_model,
            output_dir=output_dir,
            method=method,
            config=config,
        )
        console.print()
        display_training_result(result)

        # Ask about export
        console.print()
        if Confirm.ask("Export model to another format?", default=False):
            export_interactive(result.output_dir / "final")

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Training failed: {e}")
        raise typer.Exit(1)


@app.command()
def analyze(
    dataset: Annotated[
        Path,
        typer.Argument(help="Path to JSONL dataset"),
    ] = None,
) -> None:
    """Analyze a dataset and get training suggestions."""
    show_logo()
    console.print(Panel.fit("[bold]Dataset Analysis[/bold]", border_style="blue"))
    console.print()

    # Get dataset path interactively if not provided
    if dataset is None:
        dataset_str = Prompt.ask("Dataset path (JSONL file)")
        dataset = Path(dataset_str)

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

    # Get suggestions
    console.print()
    if Confirm.ask("Get training parameter suggestions?", default=True):
        available_vram = get_available_vram() or 8
        suggestions = get_suggestions(stats, available_vram, use_ai=True)
        is_ai = "AI" in suggestions.reasoning if suggestions.reasoning else False
        console.print()
        display_suggestions(suggestions, is_ai)


@app.command()
def export(
    model_path: Annotated[
        Path,
        typer.Argument(help="Path to trained model directory"),
    ] = None,
) -> None:
    """Export trained model to different formats."""
    show_logo()
    console.print(Panel.fit("[bold]Model Export[/bold]", border_style="blue"))
    console.print()

    # Get model path interactively if not provided
    if model_path is None:
        model_str = Prompt.ask("Trained model path")
        model_path = Path(model_str)

    if not model_path.exists():
        print_error(f"Model path not found: {model_path}")
        raise typer.Exit(1)

    export_interactive(model_path)


def export_interactive(model_path: Path) -> None:
    """Interactive export flow."""
    console.print(f"Model: {model_path}")
    console.print()

    # Choose format
    console.print("Export formats:")
    console.print("  lora   - LoRA adapters only (~50-200MB)")
    console.print("  merged - Full merged model (large)")
    console.print("  gguf   - GGUF for llama.cpp/Ollama")
    console.print()

    format_choice = Prompt.ask(
        "Export format",
        choices=["lora", "merged", "gguf"],
        default="lora",
    )

    # Get output path
    if format_choice == "gguf":
        default_output = model_path.parent / f"{model_path.stem}.gguf"
    else:
        default_output = model_path.parent / f"{model_path.stem}-{format_choice}"

    output_str = Prompt.ask("Output path", default=str(default_output))
    output_path = Path(output_str)

    # GGUF quantization
    quantization = None
    if format_choice == "gguf":
        console.print()
        display_gguf_quantizations()
        console.print()
        quantization = Prompt.ask(
            "Quantization type",
            default="q4_k_m",
        )
        if quantization not in GGUF_QUANTIZATIONS:
            print_error(f"Invalid quantization: {quantization}")
            raise typer.Exit(1)

    # Confirm and export
    console.print()
    if not Confirm.ask("Start export?", default=True):
        console.print("Export cancelled.")
        return

    try:
        result = export_model(
            model_path=model_path,
            output_path=output_path,
            format=format_choice,
            quantization=quantization,
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
    show_logo()
    display_model_families()


@models_app.command("search")
def models_search(
    query: Annotated[
        str,
        typer.Argument(help="Search query"),
    ] = None,
) -> None:
    """Search HuggingFace models."""
    show_logo()

    # Get query interactively if not provided
    if query is None:
        query = Prompt.ask("Search query (or press Enter for all)", default="")

    family = None
    if Confirm.ask("Filter by model family?", default=False):
        family = Prompt.ask(
            "Family",
            choices=["llama", "mistral", "qwen", "phi"],
        )

    include_base = Confirm.ask("Include base models (not just instruct)?", default=False)

    console.print()
    console.print(f"[bold]Searching for: {query or 'all models'}[/bold]")
    console.print()

    models = search_models(
        query=query,
        family=family,
        instruct_only=not include_base,
        limit=20,
    )

    display_model_list(models)


@models_app.command("info")
def models_info(
    model_id: Annotated[
        str,
        typer.Argument(help="HuggingFace model ID"),
    ] = None,
) -> None:
    """Show detailed model information."""
    show_logo()

    # Get model ID interactively if not provided
    if model_id is None:
        model_id = Prompt.ask("Model ID (e.g., meta-llama/Llama-3.2-3B-Instruct)")

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


if __name__ == "__main__":
    app()
