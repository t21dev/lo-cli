"""Tests for dataset analyzer."""

import json
import tempfile
from pathlib import Path

import pytest

from locli.analyzer import (
    DatasetStats,
    TrainingSuggestions,
    analyze_dataset,
    count_tokens_approx,
    detect_format,
    get_sample_text,
    get_static_suggestions,
    validate_jsonl,
)


class TestCountTokensApprox:
    """Tests for approximate token counting."""

    def test_empty_string(self):
        """Test empty string returns 1."""
        assert count_tokens_approx("") == 1

    def test_short_string(self):
        """Test short string token count."""
        result = count_tokens_approx("Hello world")
        assert 2 <= result <= 4

    def test_long_string(self):
        """Test longer string token count."""
        text = "This is a longer sentence with multiple words."
        result = count_tokens_approx(text)
        assert 10 <= result <= 15


class TestDetectFormat:
    """Tests for format detection."""

    def test_chat_format(self):
        """Test chat format detection."""
        sample = {"messages": [{"role": "user", "content": "Hi"}]}
        assert detect_format(sample) == "chat"

    def test_instruction_format(self):
        """Test instruction format detection."""
        sample = {"instruction": "Write code", "output": "print('hello')"}
        assert detect_format(sample) == "instruction"

    def test_completion_format_prompt(self):
        """Test completion format with prompt/completion."""
        sample = {"prompt": "Hello", "completion": "World"}
        assert detect_format(sample) == "completion"

    def test_completion_format_text(self):
        """Test completion format with text field."""
        sample = {"text": "Hello world"}
        assert detect_format(sample) == "completion"

    def test_unknown_format(self):
        """Test unknown format detection."""
        sample = {"random_field": "value"}
        assert detect_format(sample) == "unknown"


class TestGetSampleText:
    """Tests for extracting text from samples."""

    def test_chat_format(self):
        """Test text extraction from chat format."""
        sample = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        text = get_sample_text(sample)
        assert "Hello" in text
        assert "Hi there" in text

    def test_instruction_format(self):
        """Test text extraction from instruction format."""
        sample = {
            "instruction": "Write a greeting",
            "output": "Hello!",
        }
        text = get_sample_text(sample)
        assert "Write a greeting" in text
        assert "Hello!" in text


class TestAnalyzeDataset:
    """Tests for dataset analysis."""

    def test_analyze_chat_dataset(self):
        """Test analyzing a chat format dataset."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            samples = [
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "How are you?"},
                        {"role": "assistant", "content": "I'm doing well, thanks!"},
                    ]
                },
            ]
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
            f.flush()

            stats = analyze_dataset(Path(f.name))
            assert stats.total_samples == 2
            assert stats.format_type == "chat"
            assert stats.avg_tokens > 0

    def test_analyze_instruction_dataset(self):
        """Test analyzing an instruction format dataset."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            samples = [
                {"instruction": "Say hello", "output": "Hello!"},
                {"instruction": "Say goodbye", "output": "Goodbye!"},
                {"instruction": "Count to three", "output": "1, 2, 3"},
            ]
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
            f.flush()

            stats = analyze_dataset(Path(f.name))
            assert stats.total_samples == 3
            assert stats.format_type == "instruction"


class TestValidateJsonl:
    """Tests for JSONL validation."""

    def test_valid_jsonl(self):
        """Test validating a valid JSONL file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            samples = [
                {"messages": [{"role": "user", "content": "Hi"}]},
                {"messages": [{"role": "user", "content": "Hello"}]},
            ]
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
            f.flush()

            valid, msg = validate_jsonl(Path(f.name))
            assert valid is True
            assert "2 samples" in msg

    def test_invalid_json(self):
        """Test validating a file with invalid JSON."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write("not valid json\n")
            f.write('{"valid": "json"}\n')
            f.flush()

            valid, msg = validate_jsonl(Path(f.name))
            # Should still return True if some valid samples found
            assert "Invalid JSON" in msg or valid is False

    def test_nonexistent_file(self):
        """Test validating a non-existent file."""
        valid, msg = validate_jsonl(Path("/nonexistent/file.jsonl"))
        assert valid is False
        assert "not found" in msg.lower()


class TestGetStaticSuggestions:
    """Tests for static training suggestions."""

    def test_small_dataset(self):
        """Test suggestions for a small dataset."""
        stats = DatasetStats(
            total_samples=50,
            avg_tokens=100,
            min_tokens=50,
            max_tokens=150,
            median_tokens=100,
            std_tokens=20,
            format_type="chat",
            has_system_prompt=False,
            unique_system_prompts=0,
            estimated_training_time_minutes=1.0,
        )
        suggestions = get_static_suggestions(stats, available_vram=16)

        assert suggestions.r == 8  # Small dataset = small rank
        assert suggestions.num_epochs >= 3  # More epochs for small dataset

    def test_large_dataset(self):
        """Test suggestions for a large dataset."""
        stats = DatasetStats(
            total_samples=50000,
            avg_tokens=500,
            min_tokens=100,
            max_tokens=1000,
            median_tokens=450,
            std_tokens=200,
            format_type="chat",
            has_system_prompt=True,
            unique_system_prompts=1,
            estimated_training_time_minutes=100.0,
        )
        suggestions = get_static_suggestions(stats, available_vram=24)

        assert suggestions.r >= 32  # Large dataset = larger rank
        assert suggestions.num_epochs <= 2  # Fewer epochs for large dataset

    def test_vram_affects_batch_size(self):
        """Test that available VRAM affects batch size."""
        stats = DatasetStats(
            total_samples=1000,
            avg_tokens=200,
            min_tokens=100,
            max_tokens=300,
            median_tokens=200,
            std_tokens=50,
            format_type="chat",
            has_system_prompt=False,
            unique_system_prompts=0,
            estimated_training_time_minutes=10.0,
        )

        low_vram = get_static_suggestions(stats, available_vram=8)
        high_vram = get_static_suggestions(stats, available_vram=24)

        assert high_vram.batch_size >= low_vram.batch_size
