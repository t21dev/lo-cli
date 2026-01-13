"""HuggingFace model browser for LoCLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from huggingface_hub import HfApi, ModelInfo
from rich.console import Console
from rich.table import Table

from locli.config import estimate_vram
from locli.utils import extract_model_size, get_hf_token

console = Console()

# Supported model families with their HuggingFace organization/user
SUPPORTED_FAMILIES = {
    "llama": {
        "org": "meta-llama",
        "name": "Llama",
        "variants": ["Llama-3.2", "Llama-3.1", "Llama-3"],
        "notes": "Most popular, best ecosystem",
    },
    "mistral": {
        "org": "mistralai",
        "name": "Mistral",
        "variants": ["Mistral-7B", "Mixtral-8x7B"],
        "notes": "Strong performance",
    },
    "qwen": {
        "org": "Qwen",
        "name": "Qwen",
        "variants": ["Qwen2.5", "Qwen2"],
        "notes": "Good multilingual support",
    },
    "phi": {
        "org": "microsoft",
        "name": "Phi",
        "variants": ["Phi-3.5", "Phi-3"],
        "notes": "Efficient small models",
    },
}

# Keywords that indicate instruct/chat variants
INSTRUCT_KEYWORDS = ["instruct", "chat", "it", "sft"]


@dataclass
class ModelDetails:
    """Details about a model."""

    model_id: str
    family: str
    size_b: float | None
    is_instruct: bool
    vram_lora_gb: float | None
    vram_qlora_gb: float | None
    downloads: int
    likes: int
    tags: list[str]


def is_instruct_model(model_id: str, tags: list[str] | None = None) -> bool:
    """Check if a model is an instruct/chat variant."""
    model_lower = model_id.lower()

    # Check model name
    for keyword in INSTRUCT_KEYWORDS:
        if keyword in model_lower:
            return True

    # Check tags
    if tags:
        tags_lower = [t.lower() for t in tags]
        for keyword in INSTRUCT_KEYWORDS:
            if keyword in tags_lower:
                return True
        if "conversational" in tags_lower:
            return True

    return False


def get_model_family(model_id: str) -> str | None:
    """Get the model family from a model ID."""
    model_lower = model_id.lower()

    for family_key, family_info in SUPPORTED_FAMILIES.items():
        if family_info["org"].lower() in model_lower:
            return family_key
        if family_key in model_lower:
            return family_key

    return None


def search_models(
    query: str = "",
    family: str | None = None,
    instruct_only: bool = True,
    limit: int = 20,
) -> list[ModelDetails]:
    """Search for models on HuggingFace Hub.

    Args:
        query: Search query
        family: Filter by model family (llama, mistral, qwen, phi)
        instruct_only: Only return instruct/chat variants
        limit: Maximum number of results

    Returns:
        List of ModelDetails
    """
    api = HfApi(token=get_hf_token())

    # Build search parameters
    search_kwargs = {
        "task": "text-generation",
        "sort": "downloads",
        "direction": -1,
        "limit": limit * 2 if instruct_only else limit,  # Get more to filter
    }

    # Add author filter if family specified
    if family and family in SUPPORTED_FAMILIES:
        search_kwargs["author"] = SUPPORTED_FAMILIES[family]["org"]

    # Add search query
    if query:
        search_kwargs["search"] = query

    try:
        models = api.list_models(**search_kwargs)
        results = []

        for model in models:
            model_id = model.modelId

            # Check if model matches family filter
            detected_family = get_model_family(model_id)
            if family and detected_family != family:
                continue

            # Check if it's an instruct model
            is_instruct = is_instruct_model(model_id, model.tags)
            if instruct_only and not is_instruct:
                continue

            # Skip quantized versions (we do quantization ourselves)
            model_lower = model_id.lower()
            if any(q in model_lower for q in ["gguf", "gptq", "awq", "bnb", "4bit", "8bit"]):
                continue

            # Extract model size
            size_b = extract_model_size(model_id)

            # Estimate VRAM requirements
            vram_lora = estimate_vram(size_b, "lora") if size_b else None
            vram_qlora = estimate_vram(size_b, "qlora") if size_b else None

            results.append(
                ModelDetails(
                    model_id=model_id,
                    family=detected_family or "unknown",
                    size_b=size_b,
                    is_instruct=is_instruct,
                    vram_lora_gb=vram_lora,
                    vram_qlora_gb=vram_qlora,
                    downloads=model.downloads or 0,
                    likes=model.likes or 0,
                    tags=model.tags or [],
                )
            )

            if len(results) >= limit:
                break

        return results

    except Exception as e:
        console.print(f"[red]Error searching models: {e}[/red]")
        return []


def get_model_info(model_id: str) -> ModelDetails | None:
    """Get detailed information about a specific model.

    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-8B-Instruct")

    Returns:
        ModelDetails or None if not found
    """
    api = HfApi(token=get_hf_token())

    try:
        model = api.model_info(model_id)

        # Extract model size
        size_b = extract_model_size(model_id)

        # Estimate VRAM requirements
        vram_lora = estimate_vram(size_b, "lora") if size_b else None
        vram_qlora = estimate_vram(size_b, "qlora") if size_b else None

        return ModelDetails(
            model_id=model.modelId,
            family=get_model_family(model_id) or "unknown",
            size_b=size_b,
            is_instruct=is_instruct_model(model_id, model.tags),
            vram_lora_gb=vram_lora,
            vram_qlora_gb=vram_qlora,
            downloads=model.downloads or 0,
            likes=model.likes or 0,
            tags=model.tags or [],
        )

    except Exception as e:
        console.print(f"[red]Error fetching model info: {e}[/red]")
        return None


def display_model_families() -> None:
    """Display supported model families."""
    table = Table(title="Supported Model Families")
    table.add_column("Family", style="cyan", no_wrap=True)
    table.add_column("Organization", style="green")
    table.add_column("Variants", style="yellow")
    table.add_column("Notes", style="white")

    for family_key, info in SUPPORTED_FAMILIES.items():
        table.add_row(
            info["name"],
            info["org"],
            ", ".join(info["variants"]),
            info["notes"],
        )

    console.print(table)


def display_model_list(models: list[ModelDetails]) -> None:
    """Display a list of models in a table."""
    if not models:
        console.print("[yellow]No models found[/yellow]")
        return

    table = Table(title=f"Found {len(models)} models")
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Size", style="green", justify="right")
    table.add_column("VRAM (QLoRA)", style="yellow", justify="right")
    table.add_column("VRAM (LoRA)", style="yellow", justify="right")
    table.add_column("Downloads", style="blue", justify="right")
    table.add_column("Type", style="magenta")

    for model in models:
        size_str = f"{model.size_b:.1f}B" if model.size_b else "?"
        vram_qlora = f"{model.vram_qlora_gb:.0f}GB" if model.vram_qlora_gb else "?"
        vram_lora = f"{model.vram_lora_gb:.0f}GB" if model.vram_lora_gb else "?"
        downloads = f"{model.downloads:,}" if model.downloads else "0"
        model_type = "Instruct" if model.is_instruct else "Base"

        table.add_row(
            model.model_id,
            size_str,
            vram_qlora,
            vram_lora,
            downloads,
            model_type,
        )

    console.print(table)


def display_model_info(model: ModelDetails) -> None:
    """Display detailed information about a model."""
    console.print()
    console.print(f"[bold cyan]Model:[/bold cyan] {model.model_id}")
    console.print()

    table = Table(show_header=False, box=None)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Family", model.family.title())
    table.add_row("Size", f"{model.size_b:.1f}B" if model.size_b else "Unknown")
    table.add_row("Type", "Instruct/Chat" if model.is_instruct else "Base")
    table.add_row("Downloads", f"{model.downloads:,}")
    table.add_row("Likes", f"{model.likes:,}")

    console.print(table)

    if model.vram_qlora_gb or model.vram_lora_gb:
        console.print()
        console.print("[bold]VRAM Requirements:[/bold]")
        vram_table = Table(show_header=True, box=None)
        vram_table.add_column("Method", style="cyan")
        vram_table.add_column("Estimated VRAM", style="yellow")

        if model.vram_qlora_gb:
            vram_table.add_row("QLoRA (4-bit)", f"~{model.vram_qlora_gb:.0f} GB")
        if model.vram_lora_gb:
            vram_table.add_row("LoRA (full precision)", f"~{model.vram_lora_gb:.0f} GB")

        console.print(vram_table)

    if model.tags:
        console.print()
        console.print("[bold]Tags:[/bold]", ", ".join(model.tags[:10]))


def recommend_method(
    model_id: str, available_vram: float
) -> Literal["lora", "qlora"] | None:
    """Recommend training method based on available VRAM.

    Args:
        model_id: Model ID
        available_vram: Available VRAM in GB

    Returns:
        Recommended method or None if model won't fit
    """
    size_b = extract_model_size(model_id)
    if not size_b:
        return "qlora"  # Default to QLoRA if we can't determine size

    vram_lora = estimate_vram(size_b, "lora")
    vram_qlora = estimate_vram(size_b, "qlora")

    # Add some buffer (10%)
    if available_vram >= vram_lora * 1.1:
        return "lora"
    elif available_vram >= vram_qlora * 1.1:
        return "qlora"
    else:
        return None  # Model too large


def validate_model_id(model_id: str) -> bool:
    """Validate that a model ID exists on HuggingFace.

    Args:
        model_id: HuggingFace model ID

    Returns:
        True if model exists
    """
    api = HfApi(token=get_hf_token())

    try:
        api.model_info(model_id)
        return True
    except Exception:
        return False
