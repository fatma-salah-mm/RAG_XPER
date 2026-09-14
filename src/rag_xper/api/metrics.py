"""Operational metrics helpers (JSON and Prometheus formats)."""

from __future__ import annotations

import time

from rag_xper.api import state
from rag_xper.config import settings
from rag_xper.core.cache import query_cache


def collect_metrics() -> dict[str, int | float | str]:
    """Gather current process metrics as a JSON-serializable dict."""
    bm25_count = 0
    try:
        orch = state.get_orchestrator()
        if hasattr(orch._vector_store, "_bm25"):
            bm25_count = len(orch._vector_store._bm25._chunks)
    except Exception:
        pass

    cache_stats = query_cache.get_stats()
    stats = state.get_stats()
    return {
        "uptime_seconds": int(time.time() - state.get_start_time()),
        "total_queries": stats["total_queries"],
        "total_ingests": stats["total_ingests"],
        "total_errors": stats["total_errors"],
        "bm25_indexed_chunks": bm25_count,
        "cache_hits": cache_stats["hits"],
        "cache_misses": cache_stats["misses"],
        "cache_size": cache_stats["size"],
        "vector_store_type": settings.vector_store_type,
    }


def render_prometheus(metrics: dict[str, int | float | str]) -> str:
    """Render metrics in Prometheus text exposition format."""
    numeric_keys = (
        "uptime_seconds",
        "total_queries",
        "total_ingests",
        "total_errors",
        "bm25_indexed_chunks",
        "cache_hits",
        "cache_misses",
        "cache_size",
    )
    lines: list[str] = []
    for key in numeric_keys:
        value = metrics[key]
        metric = f"rag_xper_{key}"
        lines.append(f"# HELP {metric} RAG_XPER {key}")
        lines.append(f"# TYPE {metric} gauge")
        lines.append(f"{metric} {value}")
    lines.append(f'rag_xper_vector_store_type{{type="{metrics["vector_store_type"]}"}} 1')
    return "\n".join(lines) + "\n"
