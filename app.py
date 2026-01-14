#!/usr/bin/env python3
"""LoCLI - Fine-tune LLMs locally with AI-optimized defaults."""

import sys
from pathlib import Path

# Add src to path so we can import locli
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from locli.cli import app

if __name__ == "__main__":
    app()
