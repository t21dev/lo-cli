"""Utility functions for LoCLI."""

from __future__ import annotations

import os
import platform
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

if TYPE_CHECKING:
    from rich.progress import TaskID

console = Console()


@dataclass
class GPUInfo:
    """GPU information."""

    name: str
    vram_total_gb: float
    vram_free_gb: float
    cuda_version: str | None
    driver_version: str | None
    compute_capability: tuple[int, int] | None


@dataclass
class SystemInfo:
    """System information."""

    platform: str
    python_version: str
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    cuda_available: bool
    gpus: list[GPUInfo]


def get_gpu_info() -> list[GPUInfo]:
    """Get information about available GPUs."""
    gpus = []

    try:
        import torch

        if not torch.cuda.is_available():
            return gpus

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # Convert to GB

            # Get free memory
            torch.cuda.set_device(i)
            free_memory = (
                torch.cuda.get_device_properties(i).total_memory
                - torch.cuda.memory_allocated(i)
            ) / (1024**3)

            # Get CUDA version
            cuda_version = torch.version.cuda

            # Get compute capability
            compute_capability = (props.major, props.minor)

            gpus.append(
                GPUInfo(
                    name=props.name,
                    vram_total_gb=round(total_memory, 2),
                    vram_free_gb=round(free_memory, 2),
                    cuda_version=cuda_version,
                    driver_version=None,  # Not easily accessible via PyTorch
                    compute_capability=compute_capability,
                )
            )

    except ImportError:
        pass
    except Exception:
        pass

    return gpus


def get_system_info() -> SystemInfo:
    """Get system information."""
    import psutil

    # Get RAM info
    memory = psutil.virtual_memory()
    ram_total = memory.total / (1024**3)
    ram_available = memory.available / (1024**3)

    # Get disk info
    disk = shutil.disk_usage(Path.cwd())
    disk_free = disk.free / (1024**3)

    # Check CUDA availability
    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    # Get GPU info
    gpus = get_gpu_info()

    return SystemInfo(
        platform=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        ram_total_gb=round(ram_total, 2),
        ram_available_gb=round(ram_available, 2),
        disk_free_gb=round(disk_free, 2),
        cuda_available=cuda_available,
        gpus=gpus,
    )


def display_system_info(info: SystemInfo) -> None:
    """Display system information in a formatted table."""
    table = Table(title="System Information", show_header=False, box=None)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Platform", info.platform)
    table.add_row("Python", info.python_version)
    table.add_row("RAM Total", f"{info.ram_total_gb:.1f} GB")
    table.add_row("RAM Available", f"{info.ram_available_gb:.1f} GB")
    table.add_row("Disk Free", f"{info.disk_free_gb:.1f} GB")
    table.add_row("CUDA Available", "Yes" if info.cuda_available else "No")

    console.print(table)

    if info.gpus:
        console.print()
        gpu_table = Table(title="GPU Information", show_header=True)
        gpu_table.add_column("GPU", style="cyan")
        gpu_table.add_column("Name", style="green")
        gpu_table.add_column("VRAM Total", style="yellow")
        gpu_table.add_column("VRAM Free", style="yellow")
        gpu_table.add_column("CUDA", style="magenta")
        gpu_table.add_column("Compute", style="blue")

        for i, gpu in enumerate(info.gpus):
            compute = (
                f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
                if gpu.compute_capability
                else "N/A"
            )
            gpu_table.add_row(
                str(i),
                gpu.name,
                f"{gpu.vram_total_gb:.1f} GB",
                f"{gpu.vram_free_gb:.1f} GB",
                gpu.cuda_version or "N/A",
                compute,
            )

        console.print(gpu_table)
    else:
        console.print("[yellow]No NVIDIA GPUs detected[/yellow]")


def get_available_vram() -> float:
    """Get the available VRAM in GB."""
    gpus = get_gpu_info()
    if not gpus:
        return 0.0
    return max(gpu.vram_free_gb for gpu in gpus)


def get_total_vram() -> float:
    """Get the total VRAM in GB."""
    gpus = get_gpu_info()
    if not gpus:
        return 0.0
    return max(gpu.vram_total_gb for gpu in gpus)


def check_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def create_training_progress() -> Progress:
    """Create a progress bar for training."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def create_simple_progress() -> Progress:
    """Create a simple progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    )


def extract_model_size(model_name: str) -> float | None:
    """Extract model size in billions from model name.

    Args:
        model_name: Model name/ID (e.g., "meta-llama/Llama-3.2-8B-Instruct")

    Returns:
        Model size in billions, or None if not found
    """
    # Common patterns for model sizes
    patterns = [
        r"(\d+(?:\.\d+)?)[Bb]",  # 8B, 7B, 3.5B
        r"-(\d+(?:\.\d+)?)-",  # -8- (sometimes used)
        r"(\d+(?:\.\d+)?)b(?:illion)?",  # 8billion
    ]

    for pattern in patterns:
        match = re.search(pattern, model_name, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def format_number(num: int) -> str:
    """Format a large number with commas."""
    return f"{num:,}"


def validate_dataset_path(path: str | Path) -> Path:
    """Validate that a dataset path exists and is a JSONL file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if not path.is_file():
        raise ValueError(f"Dataset path is not a file: {path}")

    if path.suffix.lower() not in [".jsonl", ".json"]:
        raise ValueError(f"Dataset must be a JSONL file, got: {path.suffix}")

    return path


def validate_output_dir(path: str | Path, create: bool = True) -> Path:
    """Validate output directory path."""
    path = Path(path)

    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {path}")

    if create and not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    return path


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]Success:[/bold green] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[bold blue]Info:[/bold blue] {message}")


def print_panel(content: str, title: str = "", style: str = "blue") -> None:
    """Print content in a panel."""
    console.print(Panel(content, title=title, border_style=style))


def confirm(message: str, default: bool = False) -> bool:
    """Ask for user confirmation."""
    from rich.prompt import Confirm

    return Confirm.ask(message, default=default)


def prompt(message: str, default: str = "") -> str:
    """Prompt for user input."""
    from rich.prompt import Prompt

    return Prompt.ask(message, default=default)


def select(message: str, choices: list[str], default: str | None = None) -> str:
    """Prompt for selection from choices."""
    from rich.prompt import Prompt

    choices_str = ", ".join(choices)
    full_message = f"{message} [{choices_str}]"
    return Prompt.ask(full_message, choices=choices, default=default)


def get_hf_token() -> str | None:
    """Get HuggingFace token from environment or config."""
    # Check environment variable
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    # Check HuggingFace CLI login
    try:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
        if token:
            return token
    except ImportError:
        pass

    return None


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
