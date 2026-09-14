"""Deprecated: use `rag-xper` instead."""
import warnings

warnings.warn(
    "main.py is deprecated. Use the console script: rag-xper",
    DeprecationWarning,
    stacklevel=1,
)

from rag_xper.cli.main import main

if __name__ == "__main__":
    main()
