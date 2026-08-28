"""
core/retrieval/base_vector_store.py for RAG_XPER

Abstract Base Class defining the interface for vector databases (Qdrant & ChromaDB).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.models import Chunk, RetrievedChunk


class BaseVectorStore(ABC):
    """Abstract interface for vector database implementations."""

    @abstractmethod
    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        """Embed and upsert a batch of chunks into the database."""
        raise NotImplementedError

    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        """Perform dense semantic similarity search."""
        raise NotImplementedError

    @abstractmethod
    def hybrid_search(
        self,
        query: str,
        top_k: int = 6,
        fetch_k: int = 25,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> List[RetrievedChunk]:
        """Perform hybrid search combining BM25 and dense vector retrieval with RRF."""
        raise NotImplementedError

    @abstractmethod
    def is_file_ingested(self, file_path: str) -> bool:
        """Check if a file has already been ingested into the store."""
        raise NotImplementedError

    @abstractmethod
    def delete_file(self, file_path: str) -> int:
        """Delete all chunks belonging to a specific source file."""
        raise NotImplementedError
