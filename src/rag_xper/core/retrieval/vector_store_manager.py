"""
rag_xper.core.retrieval.vector_store_manager

ChromaDB Vector Store Manager and Unified VectorStoreFactory for RAG_XPER.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional

from rag_xper.core.exceptions import (
    ConfigurationError,
    EmbeddingGenerationError,
    VectorDBConnectionError,
)
from rag_xper.core.models import Chunk, RetrievedChunk
from rag_xper.core.retrieval.base_vector_store import BaseVectorStore
from rag_xper.core.retrieval.bm25_retriever import BM25Retriever
from rag_xper.core.retrieval.hybrid_fusion import reciprocal_rank_fusion
from rag_xper.core.retrieval.qdrant_store_manager import QdrantStoreManager
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


class ChromaVectorStoreManager(BaseVectorStore):
    """ChromaDB implementation of BaseVectorStore."""

    def __init__(
        self,
        persist_directory: str = "./storage/chroma_db",
        collection_name: str = "rag_xper_documents",
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ) -> None:
        if not _CHROMA_AVAILABLE:
            raise VectorDBConnectionError("chromadb is not installed. Please install chromadb.")

        self._persist_dir = persist_directory
        self._collection_name = collection_name
        self._embedding_fn = embedding_fn

        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        bm25_path = os.path.join(persist_directory, "bm25_index.pkl")
        self._bm25 = BM25Retriever(persist_path=bm25_path)

        if not self._bm25._chunks:
            self._sync_bm25_from_chroma()

    def _sync_bm25_from_chroma(self) -> None:
        try:
            records = self._collection.get(include=["documents", "metadatas"])
            docs = records.get("documents") or []
            ids = records.get("ids") or []
            metas = records.get("metadatas") or []

            chunks = [
                Chunk(chunk_id=cid, text=doc, metadata=meta or {})
                for cid, doc, meta in zip(ids, docs, metas)
            ]
            if chunks:
                self._bm25.add_chunks(chunks)
                logger.info("Synced %d chunks from ChromaDB into BM25 index", len(chunks))
        except Exception as exc:
            logger.warning("Could not sync BM25 from ChromaDB: %s", exc)

    def is_file_ingested(self, file_path: str, content_hash: Optional[str] = None) -> bool:
        try:
            norm_name = Path(file_path).name.lower()
            if content_hash:
                results = self._collection.get(where={"content_hash": content_hash}, limit=1)
                if results and results.get("ids"):
                    return True

            results = self._collection.get(where={"source": file_path}, limit=1)
            if results and results.get("ids"):
                return True

            for c in self._bm25._chunks:
                if Path(c.metadata.get("source", "")).name.lower() == norm_name:
                    return True
            return False
        except Exception:
            return False

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        if self._embedding_fn is None:
            raise EmbeddingGenerationError("No embedding function provided.")

        embeddings = self._embedding_fn(texts)
        self._collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        self._bm25.add_chunks(chunks)
        return len(chunks)

    def similarity_search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        if self._embedding_fn is None:
            raise EmbeddingGenerationError("No embedding function provided.")

        query_vector = self._embedding_fn([query])[0]
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[RetrievedChunk] = []
        docs = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for cid, text, meta, dist in zip(ids, docs, metas, distances):
            score = max(0.0, 1.0 - dist)
            retrieved.append(RetrievedChunk(chunk=Chunk(chunk_id=cid, text=text, metadata=meta or {}), score=score))

        return retrieved

    def hybrid_search(
        self,
        query: str,
        top_k: int = 6,
        fetch_k: int = 25,
        alpha: float = 0.5,
    ) -> List[RetrievedChunk]:
        bm25_hits = self._bm25.search(query, top_k=fetch_k)

        query_vector = self._embedding_fn([query])[0]
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        dense_hits = []
        docs = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for rank, (cid, text, meta, dist) in enumerate(zip(ids, docs, metas, distances)):
            score = max(0.0, 1.0 - dist)
            dense_hits.append((Chunk(chunk_id=cid, text=text, metadata=meta or {}), score, rank))

        return reciprocal_rank_fusion(dense_hits=dense_hits, bm25_hits=bm25_hits, top_k=top_k, alpha=alpha)

    def delete_file(self, file_path: str) -> int:
        self._collection.delete(where={"source": file_path})
        return self._bm25.remove_file_chunks(file_path)


class VectorStoreFactory:
    """Factory creating BaseVectorStore instances based on configuration."""

    @staticmethod
    def create_vector_store(
        config,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ) -> BaseVectorStore:
        store_type = getattr(config, "vector_store_type", "qdrant").lower()

        if store_type == "qdrant":
            return QdrantStoreManager(
                collection_name=getattr(config, "collection_name", "rag_xper_documents"),
                storage_path=getattr(config, "qdrant_storage_path", "./storage/qdrant_db"),
                url=getattr(config, "qdrant_url", None),
                embedding_dim=getattr(config, "embedding_dim", 3072),
                embedding_fn=embedding_fn,
            )
        elif store_type == "chromadb":
            return ChromaVectorStoreManager(
                persist_directory=getattr(config, "vector_db_path", "./storage/chroma_db"),
                collection_name=getattr(config, "collection_name", "rag_xper_documents"),
                embedding_fn=embedding_fn,
            )
        else:
            raise ConfigurationError(f"Unsupported VECTOR_STORE_TYPE: '{store_type}'")
