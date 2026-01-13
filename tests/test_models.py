"""Tests for model browser."""

import pytest

from locli.models import (
    SUPPORTED_FAMILIES,
    get_model_family,
    is_instruct_model,
)
from locli.utils import extract_model_size


class TestSupportedFamilies:
    """Tests for supported model families."""

    def test_llama_family(self):
        """Test Llama family configuration."""
        assert "llama" in SUPPORTED_FAMILIES
        assert SUPPORTED_FAMILIES["llama"]["org"] == "meta-llama"

    def test_mistral_family(self):
        """Test Mistral family configuration."""
        assert "mistral" in SUPPORTED_FAMILIES
        assert SUPPORTED_FAMILIES["mistral"]["org"] == "mistralai"

    def test_qwen_family(self):
        """Test Qwen family configuration."""
        assert "qwen" in SUPPORTED_FAMILIES
        assert SUPPORTED_FAMILIES["qwen"]["org"] == "Qwen"

    def test_phi_family(self):
        """Test Phi family configuration."""
        assert "phi" in SUPPORTED_FAMILIES
        assert SUPPORTED_FAMILIES["phi"]["org"] == "microsoft"


class TestIsInstructModel:
    """Tests for instruct model detection."""

    def test_instruct_in_name(self):
        """Test model with 'instruct' in name."""
        assert is_instruct_model("meta-llama/Llama-3.2-8B-Instruct") is True

    def test_chat_in_name(self):
        """Test model with 'chat' in name."""
        assert is_instruct_model("Qwen/Qwen2.5-7B-Chat") is True

    def test_base_model(self):
        """Test base model without instruct keywords."""
        assert is_instruct_model("meta-llama/Llama-3.2-8B") is False

    def test_it_suffix(self):
        """Test model with 'IT' suffix."""
        assert is_instruct_model("microsoft/Phi-3-mini-4k-IT") is True

    def test_with_tags(self):
        """Test detection with tags."""
        assert is_instruct_model("model-name", tags=["instruct"]) is True
        assert is_instruct_model("model-name", tags=["conversational"]) is True
        assert is_instruct_model("model-name", tags=["base"]) is False


class TestGetModelFamily:
    """Tests for model family detection."""

    def test_llama_model(self):
        """Test Llama model family detection."""
        assert get_model_family("meta-llama/Llama-3.2-8B-Instruct") == "llama"

    def test_mistral_model(self):
        """Test Mistral model family detection."""
        assert get_model_family("mistralai/Mistral-7B-Instruct-v0.2") == "mistral"

    def test_qwen_model(self):
        """Test Qwen model family detection."""
        assert get_model_family("Qwen/Qwen2.5-7B-Instruct") == "qwen"

    def test_phi_model(self):
        """Test Phi model family detection."""
        assert get_model_family("microsoft/Phi-3-mini-4k-instruct") == "phi"

    def test_unknown_model(self):
        """Test unknown model family."""
        assert get_model_family("unknown-org/random-model") is None


class TestExtractModelSize:
    """Tests for model size extraction."""

    def test_8b_model(self):
        """Test extracting 8B model size."""
        assert extract_model_size("meta-llama/Llama-3.2-8B-Instruct") == 8.0

    def test_7b_model(self):
        """Test extracting 7B model size."""
        assert extract_model_size("mistralai/Mistral-7B-Instruct") == 7.0

    def test_70b_model(self):
        """Test extracting 70B model size."""
        assert extract_model_size("meta-llama/Llama-3.1-70B-Instruct") == 70.0

    def test_decimal_size(self):
        """Test extracting decimal model size."""
        size = extract_model_size("model-3.5B")
        assert size == 3.5

    def test_lowercase_b(self):
        """Test extracting size with lowercase 'b'."""
        assert extract_model_size("model-7b-instruct") == 7.0

    def test_no_size(self):
        """Test model without size in name."""
        assert extract_model_size("model-without-size") is None
