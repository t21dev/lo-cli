"""Model exporter for LoCLI."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from peft import PeftModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import AutoModelForCausalLM, AutoTokenizer

from locli.utils import format_size, get_hf_token, print_error, print_info, print_success

console = Console()

# GGUF quantization options
GGUF_QUANTIZATIONS = {
    "q4_0": "4-bit quantization, smallest size",
    "q4_1": "4-bit quantization, slightly better quality",
    "q4_k_m": "4-bit k-quant, medium, good balance (recommended)",
    "q4_k_s": "4-bit k-quant, small",
    "q5_0": "5-bit quantization",
    "q5_1": "5-bit quantization, better quality",
    "q5_k_m": "5-bit k-quant, medium",
    "q5_k_s": "5-bit k-quant, small",
    "q6_k": "6-bit k-quant, high quality",
    "q8_0": "8-bit quantization, highest quality",
    "f16": "16-bit float, no quantization",
    "f32": "32-bit float, full precision",
}


@dataclass
class ExportResult:
    """Result of an export operation."""

    output_path: Path
    format: str
    size_bytes: int
    quantization: str | None


def load_training_info(model_path: Path) -> dict | None:
    """Load training info from a model directory."""
    info_path = model_path / "training_info.json"
    if not info_path.exists():
        # Try parent directory
        info_path = model_path.parent / "training_info.json"

    if info_path.exists():
        with open(info_path) as f:
            return json.load(f)

    return None


def get_base_model_from_adapter(adapter_path: Path) -> str | None:
    """Extract base model ID from adapter config."""
    config_path = adapter_path / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            return config.get("base_model_name_or_path")

    # Try training info
    info = load_training_info(adapter_path)
    if info:
        return info.get("base_model")

    return None


def export_lora_adapter(
    model_path: Path,
    output_path: Path,
) -> ExportResult:
    """Export LoRA adapter weights only.

    This is the default output from training - just copies the adapter files.

    Args:
        model_path: Path to the trained model (with adapter_config.json)
        output_path: Path to save the adapter

    Returns:
        ExportResult
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    # Check if this is already just an adapter
    adapter_config = model_path / "adapter_config.json"
    if not adapter_config.exists():
        raise ValueError(f"No adapter_config.json found in {model_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy adapter files
    files_to_copy = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "training_info.json",
    ]

    total_size = 0
    for filename in files_to_copy:
        src = model_path / filename
        if src.exists():
            dst = output_path / filename
            shutil.copy2(src, dst)
            total_size += src.stat().st_size

    print_success(f"Adapter exported to: {output_path}")

    return ExportResult(
        output_path=output_path,
        format="lora",
        size_bytes=total_size,
        quantization=None,
    )


def export_merged_model(
    model_path: Path,
    output_path: Path,
    save_format: Literal["safetensors", "pytorch"] = "safetensors",
) -> ExportResult:
    """Merge LoRA adapter with base model and export.

    Args:
        model_path: Path to the trained model (with adapter)
        output_path: Path to save the merged model
        save_format: Format to save weights (safetensors or pytorch)

    Returns:
        ExportResult
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    # Get base model ID
    base_model_id = get_base_model_from_adapter(model_path)
    if not base_model_id:
        raise ValueError("Could not determine base model. Please specify --base-model.")

    print_info(f"Base model: {base_model_id}")

    token = get_hf_token()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading base model...", total=None)

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=token,
        )

        progress.update(task, description="Loading LoRA adapter...")

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, str(model_path))

        progress.update(task, description="Merging adapter with base model...")

        # Merge and unload
        model = model.merge_and_unload()

        progress.update(task, description="Saving merged model...")

        # Save merged model
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(
            str(output_path),
            safe_serialization=(save_format == "safetensors"),
        )

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        tokenizer.save_pretrained(str(output_path))

        progress.update(task, description="Merged model saved!")

    # Calculate size
    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())

    print_success(f"Merged model exported to: {output_path}")
    print_info(f"Total size: {format_size(total_size)}")

    return ExportResult(
        output_path=output_path,
        format="merged",
        size_bytes=total_size,
        quantization=None,
    )


def check_llama_cpp_available() -> bool:
    """Check if llama.cpp convert script is available."""
    try:
        # Check for llama-cpp-python
        import llama_cpp  # noqa: F401

        return True
    except ImportError:
        pass

    # Check for llama.cpp CLI tools
    result = subprocess.run(
        ["which", "llama-quantize"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def export_gguf(
    model_path: Path,
    output_path: Path,
    quantization: str = "q4_k_m",
    base_model: str | None = None,
) -> ExportResult:
    """Export model to GGUF format for llama.cpp / Ollama.

    This first merges the adapter if needed, then converts to GGUF.

    Args:
        model_path: Path to the trained model
        output_path: Path to save the GGUF file
        quantization: Quantization type (q4_k_m, q5_k_m, q8_0, etc.)
        base_model: Base model ID (required if not in training info)

    Returns:
        ExportResult
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    # Validate quantization
    if quantization not in GGUF_QUANTIZATIONS:
        raise ValueError(
            f"Invalid quantization: {quantization}. "
            f"Valid options: {', '.join(GGUF_QUANTIZATIONS.keys())}"
        )

    # Get base model
    if base_model is None:
        base_model = get_base_model_from_adapter(model_path)
    if not base_model:
        raise ValueError("Could not determine base model. Please specify --base-model.")

    print_info(f"Base model: {base_model}")
    print_info(f"Quantization: {quantization} - {GGUF_QUANTIZATIONS[quantization]}")

    # Create temporary directory for merged model
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        merged_path = temp_path / "merged"

        # Check if we need to merge (has adapter_config.json)
        adapter_config = model_path / "adapter_config.json"
        if adapter_config.exists():
            print_info("Merging LoRA adapter with base model...")
            export_merged_model(model_path, merged_path)
            source_path = merged_path
        else:
            source_path = model_path

        # Convert to GGUF using llama.cpp
        print_info("Converting to GGUF format...")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Try using llama-cpp-python for conversion
        try:
            from llama_cpp import llama_cpp

            # This is a simplified approach - in practice, you'd need
            # the full llama.cpp conversion script
            print_info("Using llama-cpp-python for conversion...")

            # For now, we'll use subprocess to call the conversion script
            # if llama.cpp is installed
            convert_script = shutil.which("convert-hf-to-gguf.py") or shutil.which("convert.py")

            if convert_script:
                result = subprocess.run(
                    [
                        "python",
                        convert_script,
                        str(source_path),
                        "--outfile",
                        str(output_path),
                        "--outtype",
                        quantization,
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    raise RuntimeError(f"GGUF conversion failed: {result.stderr}")
            else:
                raise FileNotFoundError("llama.cpp conversion script not found")

        except ImportError:
            # Fallback: inform user how to convert manually
            print_error(
                "GGUF export requires llama-cpp-python or llama.cpp tools.\n"
                "Install with: pip install llama-cpp-python\n"
                "Or clone llama.cpp and run convert-hf-to-gguf.py manually."
            )
            raise RuntimeError("GGUF conversion tools not available")

    # Get output size
    if output_path.exists():
        size = output_path.stat().st_size
        print_success(f"GGUF model exported to: {output_path}")
        print_info(f"Size: {format_size(size)}")

        return ExportResult(
            output_path=output_path,
            format="gguf",
            size_bytes=size,
            quantization=quantization,
        )
    else:
        raise RuntimeError("GGUF conversion failed - output file not created")


def export_model(
    model_path: Path,
    output_path: Path,
    format: Literal["lora", "merged", "gguf"] = "lora",
    quantization: str | None = None,
    base_model: str | None = None,
) -> ExportResult:
    """Export a trained model to the specified format.

    Args:
        model_path: Path to the trained model
        output_path: Path to save the exported model
        format: Export format (lora, merged, gguf)
        quantization: GGUF quantization type (only for gguf format)
        base_model: Base model ID (optional, for gguf format)

    Returns:
        ExportResult
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    if format == "lora":
        return export_lora_adapter(model_path, output_path)

    elif format == "merged":
        return export_merged_model(model_path, output_path)

    elif format == "gguf":
        quant = quantization or "q4_k_m"
        # Ensure .gguf extension
        if not output_path.suffix == ".gguf":
            output_path = output_path.with_suffix(".gguf")
        return export_gguf(model_path, output_path, quant, base_model)

    else:
        raise ValueError(f"Unknown format: {format}")


def display_export_result(result: ExportResult) -> None:
    """Display export result summary."""
    from rich.table import Table

    table = Table(title="Export Complete")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Format", result.format.upper())
    table.add_row("Output", str(result.output_path))
    table.add_row("Size", format_size(result.size_bytes))

    if result.quantization:
        table.add_row("Quantization", result.quantization)

    console.print(table)


def display_gguf_quantizations() -> None:
    """Display available GGUF quantization options."""
    from rich.table import Table

    table = Table(title="GGUF Quantization Options")
    table.add_column("Type", style="cyan")
    table.add_column("Description", style="white")

    for quant, desc in GGUF_QUANTIZATIONS.items():
        style = "green" if quant == "q4_k_m" else None
        if quant == "q4_k_m":
            desc += " [recommended]"
        table.add_row(quant, desc, style=style)

    console.print(table)
