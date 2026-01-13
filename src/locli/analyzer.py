"""Dataset analyzer with AI suggestions for LoCLI."""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from locli.config import LoCLIConfig, LoRAConfig, TrainingConfig, load_config

console = Console()


@dataclass
class DatasetStats:
    """Statistics about a dataset."""

    total_samples: int
    avg_tokens: float
    min_tokens: int
    max_tokens: int
    median_tokens: float
    std_tokens: float
    format_type: str  # "chat", "instruction", "completion"
    has_system_prompt: bool
    unique_system_prompts: int
    estimated_training_time_minutes: float | None


@dataclass
class TrainingSuggestions:
    """AI-suggested training parameters."""

    r: int
    lora_alpha: int
    learning_rate: float
    num_epochs: int
    batch_size: int
    max_seq_length: int
    reasoning: str


def count_tokens_approx(text: str) -> int:
    """Approximate token count (roughly 4 chars per token)."""
    return max(1, len(text) // 4)


def detect_format(sample: dict) -> str:
    """Detect the format of a dataset sample."""
    # Check for chat format (messages array)
    if "messages" in sample:
        return "chat"

    # Check for instruction format
    if "instruction" in sample or "input" in sample:
        return "instruction"

    # Check for simple prompt/completion
    if "prompt" in sample and "completion" in sample:
        return "completion"

    if "text" in sample:
        return "completion"

    return "unknown"


def get_sample_text(sample: dict) -> str:
    """Extract text from a sample for token counting."""
    format_type = detect_format(sample)

    if format_type == "chat":
        messages = sample.get("messages", [])
        return " ".join(m.get("content", "") for m in messages)

    if format_type == "instruction":
        parts = []
        if "instruction" in sample:
            parts.append(sample["instruction"])
        if "input" in sample:
            parts.append(sample["input"])
        if "output" in sample:
            parts.append(sample["output"])
        if "response" in sample:
            parts.append(sample["response"])
        return " ".join(parts)

    if format_type == "completion":
        parts = []
        if "prompt" in sample:
            parts.append(sample["prompt"])
        if "completion" in sample:
            parts.append(sample["completion"])
        if "text" in sample:
            parts.append(sample["text"])
        return " ".join(parts)

    return str(sample)


def analyze_dataset(dataset_path: Path) -> DatasetStats:
    """Analyze a JSONL dataset and compute statistics.

    Args:
        dataset_path: Path to the JSONL dataset file

    Returns:
        DatasetStats with computed statistics
    """
    samples = []
    token_counts = []
    format_type = None
    system_prompts = set()

    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
                samples.append(sample)

                # Detect format from first sample
                if format_type is None:
                    format_type = detect_format(sample)

                # Count tokens
                text = get_sample_text(sample)
                token_counts.append(count_tokens_approx(text))

                # Check for system prompts
                if "messages" in sample:
                    for msg in sample["messages"]:
                        if msg.get("role") == "system":
                            system_prompts.add(msg.get("content", ""))

            except json.JSONDecodeError:
                continue

    if not samples:
        raise ValueError(f"No valid samples found in {dataset_path}")

    # Compute statistics
    avg_tokens = statistics.mean(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    median_tokens = statistics.median(token_counts)
    std_tokens = statistics.stdev(token_counts) if len(token_counts) > 1 else 0

    # Estimate training time (rough estimate: ~10 samples/second on good GPU)
    samples_per_second = 10
    total_seconds = len(samples) * 3 / samples_per_second  # 3 epochs
    estimated_time = total_seconds / 60  # in minutes

    return DatasetStats(
        total_samples=len(samples),
        avg_tokens=avg_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        median_tokens=median_tokens,
        std_tokens=std_tokens,
        format_type=format_type or "unknown",
        has_system_prompt=len(system_prompts) > 0,
        unique_system_prompts=len(system_prompts),
        estimated_training_time_minutes=estimated_time,
    )


def get_static_suggestions(stats: DatasetStats, available_vram: float) -> TrainingSuggestions:
    """Get static (non-AI) training suggestions based on dataset stats.

    Args:
        stats: Dataset statistics
        available_vram: Available VRAM in GB

    Returns:
        TrainingSuggestions
    """
    # Determine LoRA rank based on dataset complexity
    if stats.total_samples < 100:
        r = 8
    elif stats.total_samples < 1000:
        r = 16
    elif stats.total_samples < 10000:
        r = 32
    else:
        r = 64

    lora_alpha = r * 2

    # Determine learning rate based on dataset size
    if stats.total_samples < 500:
        lr = 2e-4
    elif stats.total_samples < 5000:
        lr = 1e-4
    else:
        lr = 5e-5

    # Determine epochs based on dataset size
    if stats.total_samples < 100:
        epochs = 5
    elif stats.total_samples < 1000:
        epochs = 3
    else:
        epochs = 1

    # Determine batch size based on VRAM and sequence length
    if available_vram >= 24:
        batch_size = 8
    elif available_vram >= 16:
        batch_size = 4
    elif available_vram >= 8:
        batch_size = 2
    else:
        batch_size = 1

    # Determine max sequence length
    # Use 95th percentile of token counts, rounded up to power of 2
    target_length = int(stats.median_tokens * 1.5)
    max_seq_length = min(4096, max(512, 2 ** (target_length.bit_length())))

    reasoning = f"""Based on dataset analysis:
- {stats.total_samples} samples suggests {'smaller' if stats.total_samples < 1000 else 'larger'} rank (r={r})
- Learning rate {lr} appropriate for dataset size
- {epochs} epoch(s) to avoid overfitting on {'small' if stats.total_samples < 1000 else 'large'} dataset
- Batch size {batch_size} based on ~{available_vram:.0f}GB VRAM
- Max sequence length {max_seq_length} covers {stats.median_tokens:.0f} median tokens"""

    return TrainingSuggestions(
        r=r,
        lora_alpha=lora_alpha,
        learning_rate=lr,
        num_epochs=epochs,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        reasoning=reasoning,
    )


def get_ai_suggestions(
    stats: DatasetStats, available_vram: float, config: LoCLIConfig | None = None
) -> TrainingSuggestions | None:
    """Get AI-powered training suggestions using OpenAI API.

    Args:
        stats: Dataset statistics
        available_vram: Available VRAM in GB
        config: LoCLI configuration

    Returns:
        TrainingSuggestions or None if API unavailable
    """
    if config is None:
        config = load_config()

    api_key = config.openai.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        prompt = f"""You are an ML engineer optimizing LLM fine-tuning hyperparameters.

Dataset statistics:
- Total samples: {stats.total_samples}
- Average tokens per sample: {stats.avg_tokens:.0f}
- Min/Max tokens: {stats.min_tokens}/{stats.max_tokens}
- Format: {stats.format_type}
- Has system prompts: {stats.has_system_prompt}

Available VRAM: {available_vram:.0f} GB

Suggest optimal LoRA fine-tuning parameters. Return ONLY a JSON object with these fields:
- r: LoRA rank (8, 16, 32, or 64)
- lora_alpha: LoRA alpha (typically 2x rank)
- learning_rate: float (e.g., 2e-4)
- num_epochs: integer (1-10)
- batch_size: integer (1, 2, 4, or 8)
- max_seq_length: integer (512, 1024, 2048, or 4096)
- reasoning: brief explanation (1-2 sentences)

JSON only, no markdown:"""

        response = client.chat.completions.create(
            model=config.openai.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
        )

        content = response.choices[0].message.content
        if not content:
            return None

        # Parse JSON response
        # Handle potential markdown code blocks
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        data = json.loads(content)

        return TrainingSuggestions(
            r=int(data.get("r", 16)),
            lora_alpha=int(data.get("lora_alpha", 32)),
            learning_rate=float(data.get("learning_rate", 2e-4)),
            num_epochs=int(data.get("num_epochs", 3)),
            batch_size=int(data.get("batch_size", 4)),
            max_seq_length=int(data.get("max_seq_length", 2048)),
            reasoning=data.get("reasoning", "AI-suggested parameters based on dataset analysis."),
        )

    except ImportError:
        console.print("[yellow]OpenAI package not installed. Using static suggestions.[/yellow]")
        return None
    except Exception as e:
        console.print(f"[yellow]AI suggestions unavailable: {e}[/yellow]")
        return None


def get_suggestions(
    stats: DatasetStats, available_vram: float, use_ai: bool = True
) -> TrainingSuggestions:
    """Get training suggestions (AI if available, otherwise static).

    Args:
        stats: Dataset statistics
        available_vram: Available VRAM in GB
        use_ai: Whether to try using AI suggestions

    Returns:
        TrainingSuggestions
    """
    if use_ai:
        ai_suggestions = get_ai_suggestions(stats, available_vram)
        if ai_suggestions:
            return ai_suggestions

    return get_static_suggestions(stats, available_vram)


def display_stats(stats: DatasetStats) -> None:
    """Display dataset statistics in a formatted table."""
    table = Table(title="Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Total Samples", f"{stats.total_samples:,}")
    table.add_row("Format", stats.format_type.title())
    table.add_row("Average Tokens", f"{stats.avg_tokens:.0f}")
    table.add_row("Median Tokens", f"{stats.median_tokens:.0f}")
    table.add_row("Min Tokens", f"{stats.min_tokens:,}")
    table.add_row("Max Tokens", f"{stats.max_tokens:,}")
    table.add_row("Std Dev Tokens", f"{stats.std_tokens:.0f}")
    table.add_row("Has System Prompt", "Yes" if stats.has_system_prompt else "No")

    if stats.has_system_prompt:
        table.add_row("Unique System Prompts", str(stats.unique_system_prompts))

    if stats.estimated_training_time_minutes:
        if stats.estimated_training_time_minutes < 60:
            time_str = f"~{stats.estimated_training_time_minutes:.0f} minutes"
        else:
            hours = stats.estimated_training_time_minutes / 60
            time_str = f"~{hours:.1f} hours"
        table.add_row("Est. Training Time", time_str)

    console.print(table)


def display_suggestions(suggestions: TrainingSuggestions, is_ai: bool = False) -> None:
    """Display training suggestions."""
    title = "AI-Suggested Parameters" if is_ai else "Suggested Parameters"

    table = Table(title=title)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_column("Description", style="white")

    table.add_row("r (rank)", str(suggestions.r), "LoRA rank - higher = more capacity")
    table.add_row("lora_alpha", str(suggestions.lora_alpha), "LoRA scaling factor")
    table.add_row("learning_rate", f"{suggestions.learning_rate:.0e}", "Optimizer learning rate")
    table.add_row("num_epochs", str(suggestions.num_epochs), "Training epochs")
    table.add_row("batch_size", str(suggestions.batch_size), "Batch size per device")
    table.add_row("max_seq_length", str(suggestions.max_seq_length), "Maximum sequence length")

    console.print(table)

    if suggestions.reasoning:
        console.print()
        console.print(Panel(suggestions.reasoning, title="Reasoning", border_style="blue"))


def suggestions_to_config(suggestions: TrainingSuggestions) -> tuple[LoRAConfig, TrainingConfig]:
    """Convert suggestions to configuration objects.

    Args:
        suggestions: Training suggestions

    Returns:
        Tuple of (LoRAConfig, TrainingConfig)
    """
    lora_config = LoRAConfig(
        r=suggestions.r,
        lora_alpha=suggestions.lora_alpha,
    )

    training_config = TrainingConfig(
        learning_rate=suggestions.learning_rate,
        num_epochs=suggestions.num_epochs,
        batch_size=suggestions.batch_size,
        max_seq_length=suggestions.max_seq_length,
    )

    return lora_config, training_config


def validate_jsonl(dataset_path: Path) -> tuple[bool, str]:
    """Validate a JSONL file format.

    Args:
        dataset_path: Path to the JSONL file

    Returns:
        Tuple of (is_valid, message)
    """
    if not dataset_path.exists():
        return False, f"File not found: {dataset_path}"

    if dataset_path.suffix.lower() not in [".jsonl", ".json"]:
        return False, f"Expected .jsonl or .json file, got: {dataset_path.suffix}"

    valid_count = 0
    error_count = 0
    first_error = None

    with open(dataset_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
                format_type = detect_format(sample)
                if format_type == "unknown":
                    if first_error is None:
                        first_error = f"Line {i}: Unknown format - expected 'messages', 'instruction/output', or 'prompt/completion'"
                    error_count += 1
                else:
                    valid_count += 1
            except json.JSONDecodeError as e:
                if first_error is None:
                    first_error = f"Line {i}: Invalid JSON - {e}"
                error_count += 1

    if valid_count == 0:
        return False, first_error or "No valid samples found"

    if error_count > 0:
        return True, f"Found {valid_count} valid samples, {error_count} invalid. First error: {first_error}"

    return True, f"Valid JSONL with {valid_count} samples"
