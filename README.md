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
```

### Optional Dependencies

```bash
# For AI-powered suggestions (requires OpenAI API key)
pip install openai

# For GGUF export
pip install llama-cpp-python
```

> **Note:** The default OpenAI model is `gpt-4.1-mini`. Make sure you have access to this model enabled in your [OpenAI developer account](https://platform.openai.com/settings/organization/limits). You can change the model in `.env` by setting `OPENAI_MODEL=gpt-4o` or another supported model.

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
python app.py train
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
python app.py train           # Training wizard
python app.py analyze         # Analyze dataset & get suggestions
python app.py export          # Export model (LoRA/merged/GGUF)
python app.py models list     # List supported model families
python app.py models search   # Search HuggingFace models
python app.py models info     # Show model details & VRAM requirements
python app.py info            # Check GPU, VRAM, CUDA status
```

## Hardware Requirements

| Model Size | Method | Min VRAM |
|------------|--------|----------|
| 3B | QLoRA | 4GB |
| 3B | LoRA | 8GB |
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
- 4GB+ VRAM (QLoRA with 3B models) / 6GB+ for 7B models

### PyTorch with CUDA

The default `pip install` may install CPU-only PyTorch. For GPU training, install PyTorch with CUDA:

```bash
# Check if CUDA is working
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

For RTX 40/50 series GPUs, use CUDA 12.4. For older GPUs, use `cu118` or `cu121`.

### HuggingFace Authentication (for Llama, Mistral, etc.)

Gated models require HuggingFace authentication. If you get `401` or `403` errors:

- **403 Forbidden**: Token exists but lacks permissions → Create new token with read access
- **401 Unauthorized**: Token invalid/missing → Re-run `huggingface-cli login`

**Step 1: Create a Fine-Grained Token**

Go to https://huggingface.co/settings/tokens and create a new token:
- Select "Fine-grained token"
- Name it (e.g., "llama-access")
- Under Permissions, select: **Read access to contents of all public gated repos you can access**
- Click "Create token"
- Copy the token (starts with `hf_`)

**Step 2: Accept Meta's License**

Visit https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct and click **"Agree and access repository"** button.

**Step 3: Login with Your Token**

```bash
huggingface-cli login
# When prompted, paste your token (it won't show as you type)
```

**Step 4: Verify Login**

```bash
huggingface-cli whoami
# Should show your HF username
```

**Step 5: Add Token to .env**

```bash
HF_TOKEN=hf_your_token_here
```

**Verification Checklist:**
- [ ] Visited meta-llama repo page and clicked "Agree"
- [ ] Generated fine-grained token with gated repo read access
- [ ] Ran `huggingface-cli login` with new token
- [ ] Confirmed `huggingface-cli whoami` shows your username
- [ ] Set `HF_TOKEN` in `.env` file
- [ ] If still failing, try clearing cache: `rm -rf ~/.cache/huggingface/`

## Development

```bash
# Run tests
pip install pytest
pytest tests/ -v

# Run linting
pip install ruff
ruff check src tests
```

## License

MIT License - see [LICENSE](LICENSE)

## Author

Created by [@TriptoAfsin](https://github.com/TriptoAfsin) | [t21.dev](https://github.com/t21dev)
