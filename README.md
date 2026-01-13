# LoCLI

> Fine-tune LLMs locally with AI-optimized defaults

LoCLI makes fine-tuning LLMs accessible to developers. Just point it at your dataset and go.

## Features

- **Multiple Model Families** - Llama, Mistral, Qwen, Phi from HuggingFace
- **LoRA & QLoRA** - Fine-tune on consumer GPUs (6GB+ VRAM)
- **AI-Optimized Defaults** - Analyzes your dataset and suggests hyperparameters
- **Interactive Mode** - Guided setup for beginners
- **Export Options** - LoRA adapters, merged models, GGUF for Ollama

## Installation

```bash
# Clone the repository
git clone https://github.com/t21dev/locli.git
cd locli

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package locally
pip install -e .
```

### Optional Dependencies

```bash
# For AI-powered suggestions (requires OpenAI API key)
pip install openai

# For GGUF export
pip install llama-cpp-python
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
# For gated models (Llama, etc.)
HF_TOKEN=hf_your_token_here

# For AI-powered suggestions (optional)
OPENAI_API_KEY=sk-your_key_here
```

### Config File (Optional)

Create `locli.yaml` for custom defaults:

```yaml
lora:
  r: 16
  lora_alpha: 32

training:
  learning_rate: 2e-4
  num_epochs: 3
  batch_size: 4
```

## Quick Start

```bash
# Interactive mode (recommended for first time)
locli train --interactive

# Or specify everything
locli train \
  --dataset data.jsonl \
  --base-model meta-llama/Llama-3.2-8B-Instruct \
  --method qlora
```

## Hardware Requirements

| Model Size | Method | Min VRAM |
|------------|--------|----------|
| 7B | QLoRA | 6GB |
| 7B | LoRA | 14GB |
| 13B | QLoRA | 10GB |

## Commands

```bash
# Training
locli train -i                    # Interactive training wizard
locli train -d data.jsonl -m meta-llama/Llama-3.2-8B-Instruct

# Model browsing
locli models list                 # List supported model families
locli models search "llama 8b"    # Search HuggingFace models
locli models info <model-id>      # Show model details + VRAM requirements

# Dataset analysis
locli analyze data.jsonl          # Analyze dataset statistics
locli analyze data.jsonl --suggest  # Get AI-powered training suggestions

# Export
locli export ./output/final --format lora    # Export LoRA adapters
locli export ./output/final --format merged  # Merge with base model
locli export ./output/final --format gguf    # Convert to GGUF

# System info
locli info                        # Check GPU, VRAM, CUDA status
```

## Dataset Format

LoCLI supports JSONL files with these formats:

**Chat format (recommended):**
```json
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}
```

**Instruction format:**
```json
{"instruction": "Write a greeting", "output": "Hello!"}
```

**Completion format:**
```json
{"prompt": "Hello", "completion": "World"}
```

A sample dataset is included: `sample.jsonl`

## Training Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset, -d` | Path to JSONL dataset | Required |
| `--base-model, -m` | HuggingFace model ID | Required |
| `--method` | `lora` or `qlora` | `qlora` |
| `--output, -o` | Output directory | `./output` |
| `--epochs` | Number of epochs | `3` |
| `--batch-size` | Batch size | `4` |
| `--lr` | Learning rate | `2e-4` |
| `--r` | LoRA rank | `16` |
| `--resume` | Resume from checkpoint | - |
| `--interactive, -i` | Interactive mode | `false` |

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- 6GB+ VRAM (QLoRA with 7B models)

## Development

```bash
# Install dev dependencies
pip install pytest pytest-cov ruff

# Run tests
pytest tests/ -v

# Run linting
ruff check src tests
```

## License

MIT License - see [LICENSE](LICENSE)

## Author

Created by [@TriptoAfsin](https://github.com/TriptoAfsin) | [t21dev](https://github.com/t21dev)
