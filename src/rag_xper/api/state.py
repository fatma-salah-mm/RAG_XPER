"""Process-wide API state shared across route modules."""

from __future__ import annotations

import time

_start_time = time.time()
_stats = {"total_queries": 0, "total_ingests": 0, "total_errors": 0}
_orchestrator = None


def get_start_time() -> float:
    return _start_time


def get_stats() -> dict:
    return _stats


def get_orchestrator():
    """Lazily load pipeline components once per worker process."""
    global _orchestrator
    if _orchestrator is None:
        from rag_xper.bootstrap import build_orchestrator

        _orchestrator = build_orchestrator()
    return _orchestrator


def reset_orchestrator() -> None:
    """Reset the cached orchestrator (used in tests)."""
    global _orchestrator
    _orchestrator = None


def close_resources() -> None:
    """Release vector-store clients and cached orchestrator on shutdown."""
    global _orchestrator
    if _orchestrator is None:
        return
    vector_store = _orchestrator._vector_store
    client = getattr(vector_store, "_client", None)
    if client is not None and hasattr(client, "close"):
        try:
            client.close()
        except Exception:
            pass
    _orchestrator = None
