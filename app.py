"""Deprecated: use `rag-xper-ui` instead."""
import warnings

warnings.warn(
    "app.py is deprecated. Use the console script: rag-xper-ui",
    DeprecationWarning,
    stacklevel=1,
)

from rag_xper.ui.gradio_app import run_ui

if __name__ == "__main__":
    run_ui()
