from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()


# Shared client
client = anthropic.Anthropic()

# Model aliases
FAST_MODEL = os.getenv("CLAUDE_FAST_MODEL", "claude-haiku-4-5")
SMART_MODEL = os.getenv("CLAUDE_SMART_MODEL", "claude-opus-4-7")

# Token limits
DEFAULT_MAX_TOKENS = int(os.getenv("CLAUDE_DEFAULT_MAX_TOKENS", "1024"))
EXTRACTION_MAX_TOKENS = int(os.getenv("CLAUDE_EXTRACTION_MAX_TOKENS", "512"))
RANKING_MAX_TOKENS = int(os.getenv("CLAUDE_RANKING_MAX_TOKENS", "4096"))
