"""
conftest.py for RAG_XPER
"""
import os
import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Settings reads the environment at import time, so placeholders must be set before
# rag_xper.config loads. This keeps the suite hermetic: tests never reach a real
# provider, and they pass without a .env file. Exported values still take precedence.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used-for-network-calls")
os.environ.setdefault("LLM_PROVIDER", "gemini")
