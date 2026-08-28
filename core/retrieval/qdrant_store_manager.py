"""
core/retrieval/qdrant_store_manager.py for RAG_XPER

High-performance, local embedded Qdrant Vector Database implementation.
Supports:
- Local disk persistence via embedded Rust engine (`QdrantClient(path=...)`)
- Dense Cosine vector similarity search
- Integrated Arabic-aware BM25 lexical search
- Reciprocal Rank Fusion (RRF) Hybrid Search
- Deduplication and file ingestion cache
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False

from core.exceptions import EmbeddingGenerationError, VectorDBConnectionError
from core.models import Chunk, RetrievedChunk
from core.retrieval.base_vector_store import BaseVectorStore
from core.retrieval.bm25_retriever import BM25Retriever
from utils.logger import get_logger

logger = get_logger(__name__)


class QdrantStoreManager(BaseVectorStore):
    """Local embedded Qdrant vector store with BM25 hybrid search."""

    def __init__(
        self,
        storage_path: str = "./storage/qdrant_db",
        collection_name: str = "rag_xper_documents",
        embedding_dim: int = 3072,
        embedding_fn=None,
    ) -> None:
        if not _QDRANT_AVAILABLE:
            raise VectorDBConnectionError(
                "qdrant-client is not installed. Please install it with: pip install qdrant-client"
            )

        self._storage_path = storage_path
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim
        self._embedding_fn = embedding_fn
        self._bm25 = BM25Retriever()

        Path(storage_path).mkdir(parents=True, exist_ok=True)

        try:
            self._client = QdrantClient(path=storage_path)
            self._ensure_collection()
            self._sync_bm25_from_qdrant()
            logger.info(
                "QdrantStoreManager ready (storage='%s', collection='%s', bm25_docs=%d)",
                storage_path,
                collection_name,
                self._bm25.total_docs,
            )
        except Exception as exc:
            raise VectorDBConnectionError(
                f"Could not initialise Qdrant at '{storage_path}': {exc}"
            ) from exc

    def _ensure_collection(self) -> None:
        """Create collection if it does not already exist."""
        collections = self._client.get_collections().collections
        exists = any(c.name == self._collection_name for c in collections)
        if not exists:
            logger.info(
                "Creating new Qdrant collection '%s' (dim=%d, distance=COSINE)",
                self._collection_name,
                self._embedding_dim,
            )
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self._embedding_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def _sync_bm25_from_qdrant(self) -> None:
        """Load all existing documents from Qdrant into the in-memory BM25 index."""
        try:
            self._bm25.clear()
            limit = 10000
            records, _ = self._client.scroll(
                collection_name=self._collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            synced_chunks: List[Chunk] = []
            for record in records:
                payload = record.payload or {}
                text = payload.get("text", "")
                chunk_id = payload.get("chunk_id", str(record.id))
                metadata = {k: v for k, v in payload.items() if k != "text"}
                if text:
                    synced_chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=metadata))

            if synced_chunks:
                self._bm25.add_chunks(synced_chunks)
                logger.info("Synced %d chunks from Qdrant into BM25 index", len(synced_chunks))
        except Exception as exc:
            logger.warning("Could not sync BM25 index from Qdrant: %s", exc)

    def is_file_ingested(self, file_path: str) -> bool:
        """Check if any chunks from this file path already exist in Qdrant."""
        try:
            target_norm = Path(file_path).name
            records, _ = self._client.scroll(
                collection_name=self._collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                src = (r.payload or {}).get("source", "")
                if target_norm in src or Path(src).name == target_norm:
                    return True
            return False
        except Exception:
            return False

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        if self._embedding_fn is None:
            raise EmbeddingGenerationError("No embedding function provided to QdrantStoreManager.")

        try:
            embeddings = self._embedding_fn(texts)
        except Exception as exc:
            raise EmbeddingGenerationError(f"Embedding generation failed: {exc}") from exc

        if embeddings and len(embeddings[0]) != self._embedding_dim:
            self._embedding_dim = len(embeddings[0])

        points: List[qmodels.PointStruct] = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            payload = dict(chunk.metadata)
            payload["text"] = chunk.text
            payload["chunk_id"] = chunk.chunk_id

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=payload,
                )
            )

        try:
            batch_size = 64
            for idx in range(0, len(points), batch_size):
                batch = points[idx : idx + batch_size]
                self._client.upsert(collection_name=self._collection_name, points=batch)

            self._bm25.add_chunks(chunks)
            logger.info("Upserted %d chunks into Qdrant & BM25 index", len(chunks))
            return len(chunks)
        except Exception as exc:
            raise VectorDBConnectionError(f"Failed to upsert points into Qdrant: {exc}") from exc

    def _search_points(self, query_vector: List[float], limit: int):
        """Universal search helper compatible with all qdrant-client versions."""
        if hasattr(self._client, "query_points"):
            res = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            return getattr(res, "points", res)
        elif hasattr(self._client, "search"):
            return self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
        elif hasattr(self._client, "search_points"):
            res = self._client.search_points(
                collection_name=self._collection_name,
                vector=query_vector,
                limit=limit,
                with_payload=True,
            )
            return getattr(res, "points", res)
        else:
            raise AttributeError("QdrantClient has no search or query_points method.")

    def similarity_search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        if self._embedding_fn is None:
            raise EmbeddingGenerationError("No embedding function provided.")

        try:
            query_vector = self._embedding_fn([query])[0]
        except Exception as exc:
            logger.warning("Dense embedding failed (%s). Falling back to BM25.", exc)
            bm25_results = self._bm25.search(query, top_k=top_k)
            return [RetrievedChunk(chunk=c, score=s) for c, s, _ in bm25_results]

        try:
            hits = self._search_points(query_vector=query_vector, limit=top_k)
            results: List[RetrievedChunk] = []
            for hit in hits:
                payload = hit.payload or {}
                text = payload.get("text", "")
                chunk_id = payload.get("chunk_id", str(hit.id))
                metadata = {k: v for k, v in payload.items() if k != "text"}
                results.append(
                    RetrievedChunk(
                        chunk=Chunk(chunk_id=chunk_id, text=text, metadata=metadata),
                        score=float(hit.score),
                    )
                )
            return results
        except Exception as exc:
            raise VectorDBConnectionError(f"Qdrant search failed: {exc}") from exc

    def hybrid_search(
        self,
        query: str,
        top_k: int = 6,
        fetch_k: int = 25,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> List[RetrievedChunk]:
        """
        Hybrid search combining BM25 and Qdrant Dense Vector search using Reciprocal Rank Fusion (RRF).
        """
        # 1. BM25 search
        bm25_hits = self._bm25.search(query, top_k=fetch_k)

        # 2. Dense vector search
        dense_hits = []
        try:
            query_vector = self._embedding_fn([query])[0]
            hits = self._search_points(query_vector=query_vector, limit=fetch_k)
            for dense_rank, hit in enumerate(hits):
                payload = hit.payload or {}
                text = payload.get("text", "")
                chunk_id = payload.get("chunk_id", str(hit.id))
                metadata = {k: v for k, v in payload.items() if k != "text"}
                dense_hits.append(
                    (Chunk(chunk_id=chunk_id, text=text, metadata=metadata), float(hit.score), dense_rank)
                )
        except Exception as exc:
            logger.warning("Dense search failed in hybrid search (%s). Returning pure BM25.", exc)
            return [RetrievedChunk(chunk=c, score=s) for c, s, _ in bm25_hits[:top_k]]

        # 3. Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}

        dense_weight = alpha
        bm25_weight = 1.0 - alpha

        for chunk, _, rank in dense_hits:
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (dense_weight / (rrf_k + rank + 1))

        for chunk, _, rank in bm25_hits:
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (bm25_weight / (rrf_k + rank + 1))

        sorted_cids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        results: List[RetrievedChunk] = []
        for cid in sorted_cids[:top_k]:
            results.append(
                RetrievedChunk(
                    chunk=chunk_map[cid],
                    score=round(rrf_scores[cid], 4),
                )
            )

        return results

    def delete_file(self, file_path: str) -> int:
        target_name = Path(file_path).name
        try:
            records, _ = self._client.scroll(
                collection_name=self._collection_name,
                limit=10000,
                with_payload=True,
            )
            ids_to_del = []
            for r in records:
                src = (r.payload or {}).get("source", "")
                if target_name in src or Path(src).name == target_name:
                    ids_to_del.append(r.id)

            if ids_to_del:
                self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=qmodels.PointIdsList(points=ids_to_del),
                )
                self._sync_bm25_from_qdrant()
            return len(ids_to_del)
        except Exception as exc:
            logger.error("Failed to delete file '%s' from Qdrant: %s", file_path, exc)
            return 0
