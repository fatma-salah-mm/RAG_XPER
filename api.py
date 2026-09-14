"""Deprecated: use `rag-xper-api` instead."""
import warnings

warnings.warn(
    "api.py is deprecated. Use the console script: rag-xper-api",
    DeprecationWarning,
    stacklevel=1,
)

from rag_xper.api.app import run_api

if __name__ == "__main__":
    run_api()
