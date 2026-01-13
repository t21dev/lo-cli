# LoCLI

```
  ██╗      ██████╗  ██████╗██╗     ██╗
  ██║     ██╔═══██╗██╔════╝██║     ██║
  ██║     ██║   ██║██║     ██║     ██║
  ██║     ██║   ██║██║     ██║     ██║
  ███████╗╚██████╔╝╚██████╗███████╗██║
  ╚══════╝ ╚═════╝  ╚═════╝╚══════╝╚═╝

  Fine-tune LLMs locally with AI-optimized defaults
  by t21.dev
```

LoCLI makes fine-tuning LLMs accessible to developers. Just point it at your dataset and go.

## Features

- **Multiple Model Families** - Llama, Mistral, Qwen, Phi from HuggingFace
- **LoRA & QLoRA** - Fine-tune on consumer GPUs (6GB+ VRAM)
- **AI-Optimized Defaults** - Analyzes your dataset and suggests hyperparameters
- **Interactive CLI** - Guided step-by-step setup
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

## Quick Start

```bash
# Start the training wizard
locli train
```

The interactive wizard guides you through:

```
Step 1: Dataset        → Enter path, validate, show stats
Step 2: Model          → Choose HuggingFace model
Step 3: Method         → LoRA or QLoRA (auto-recommended)
Step 4: Parameters     → AI-suggested or custom
Step 5: Output         → Choose output directory
Summary               → Review and start training
```

## Commands

All commands are interactive and will prompt for required inputs:

```bash
locli train           # Training wizard
locli analyze         # Analyze dataset & get suggestions
locli export          # Export model (LoRA/merged/GGUF)
locli models list     # List supported model families
locli models search   # Search HuggingFace models
locli models info     # Show model details & VRAM requirements
locli info            # Check GPU, VRAM, CUDA status
```

## Hardware Requirements

| Model Size | Method | Min VRAM |
|------------|--------|----------|
| 7B | QLoRA | 6GB |
| 7B | LoRA | 14GB |
| 13B | QLoRA | 10GB |

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

## Config File (Optional)

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

Created by [@TriptoAfsin](https://github.com/TriptoAfsin) | [t21.dev](https://github.com/t21dev)
