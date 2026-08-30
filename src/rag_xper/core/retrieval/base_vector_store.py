"""
rag_xper.core.retrieval.base_vector_store

Abstract base class for vector store managers.
"""
from __future__ import annotations

import abc
from typing import List, Optional

from rag_xper.core.models import Chunk, RetrievedChunk


class BaseVectorStore(abc.ABC):
    """Abstract interface defining the contract for all vector store providers."""

    @abc.abstractmethod
    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        """Embed and persist a list of Chunks. Returns number of chunks stored."""
        raise NotImplementedError

    @abc.abstractmethod
    def similarity_search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        """Perform dense vector search for the given query string."""
        raise NotImplementedError

    @abc.abstractmethod
    def hybrid_search(
        self,
        query: str,
        top_k: int = 6,
        fetch_k: int = 25,
        alpha: float = 0.5,
    ) -> List[RetrievedChunk]:
        """Perform hybrid search combining dense vectors and lexical BM25."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_file_ingested(self, file_path: str, content_hash: Optional[str] = None) -> bool:
        """Return True if chunks for the given file or content hash are already indexed."""
        raise NotImplementedError

    @abc.abstractmethod
    def delete_file(self, file_path: str) -> int:
        """Delete all chunks belonging to a specific source file."""
        raise NotImplementedError
