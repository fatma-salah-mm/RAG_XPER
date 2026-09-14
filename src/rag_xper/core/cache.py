"""
rag_xper.core.cache

High-performance In-Memory Query & Result Caching layer with TTL (Time-To-Live)
and Arabic question normalization for instant (<5ms) sub-second responses.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any

from rag_xper.core.models import RAGResponse
from rag_xper.core.retrieval.bm25_retriever import normalize_arabic
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    response: RAGResponse
    created_at: float
    hits: int = 0


class QueryCache:
    """Thread-safe In-Memory LRU Cache with TTL expiration and normalization."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, CacheEntry] = {}
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._lock = threading.Lock()

    def _normalize_key(self, question: str, top_k: int = 6) -> str:
        """Create a deterministic hash from normalized Arabic/English text."""
        norm_q = normalize_arabic(question.strip().lower())
        raw_key = f"{norm_q}::top_k={top_k}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(self, question: str, top_k: int = 6) -> RAGResponse | None:
        """Retrieve cached response if valid and not expired."""
        key = self._normalize_key(question, top_k)
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                self._stats["misses"] += 1
                return None

            if time.time() - entry.created_at > self.ttl_seconds:
                del self._cache[key]
                self._stats["misses"] += 1
                self._stats["evictions"] += 1
                return None

            entry.hits += 1
            self._stats["hits"] += 1
            logger.debug("⚡ Cache HIT for question: '%s' (hits: %d)", question[:30], entry.hits)
            return entry.response

    def set(self, question: str, response: RAGResponse, top_k: int = 6) -> None:
        """Store response in cache with LRU eviction when capacity reached."""
        key = self._normalize_key(question, top_k)
        with self._lock:
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
                del self._cache[oldest_key]
                self._stats["evictions"] += 1

            self._cache[key] = CacheEntry(response=response, created_at=time.time(), hits=0)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
        logger.info("QueryCache cleared.")

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            hits = self._stats["hits"]
            misses = self._stats["misses"]
            size = len(self._cache)
            max_size = self.max_size
            ttl_seconds = self.ttl_seconds
        return {
            "size": size,
            "max_size": max_size,
            "ttl_seconds": ttl_seconds,
            "hits": hits,
            "misses": misses,
            "hit_ratio": (round(hits / (hits + misses), 3) if (hits + misses) > 0 else 0.0),
        }


# Global Query Cache instance
query_cache = QueryCache()
