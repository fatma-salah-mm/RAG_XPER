"""
rag_xper.core.retrieval.qdrant_store_manager

High-performance Qdrant Vector Store supporting both Embedded local storage and Remote Server mode,
payload-based deduplication, BM25 disk persistence, and Reciprocal Rank Fusion.
"""
from __future__ import annotations

import os
import uuid
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
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False


class QdrantStoreManager(BaseVectorStore):
    """Production Qdrant Vector Store Manager."""

    def __init__(
        self,
        collection_name: str = "rag_xper_documents",
        storage_path: Optional[str] = "./storage/qdrant_db",
        url: Optional[str] = None,
        embedding_dim: int = 3072,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ) -> None:
        if not _QDRANT_AVAILABLE:
            raise VectorDBConnectionError(
                "qdrant-client is not installed. Please install it with: pip install qdrant-client"
            )

        self._collection_name = collection_name
        self._embedding_dim = embedding_dim
        self._embedding_fn = embedding_fn

        # 1. Initialize Client (Remote Server mode or Embedded Local mode)
        if url:
            logger.info("Connecting to Qdrant Server at '%s'", url)
            self._client = QdrantClient(url=url)
        else:
            path_str = storage_path or "./storage/qdrant_db"
            Path(path_str).mkdir(parents=True, exist_ok=True)
            logger.info("Initializing Local Embedded Qdrant at '%s'", path_str)
            self._client = QdrantClient(path=path_str)

        # 2. Ensure collection exists with correct vector size
        self._ensure_collection()

        # 3. Persistent BM25 Retriever
        bm25_persist = os.path.join(storage_path or "./storage", "bm25_index.pkl")
        self._bm25 = BM25Retriever(persist_path=bm25_persist)

        # 4. Sync BM25 from Qdrant if empty
        if not self._bm25._chunks:
            self._sync_bm25_from_qdrant()

    def _ensure_collection(self) -> None:
        try:
            collections = self._client.get_collections().collections
            exists = any(c.name == self._collection_name for c in collections)

            if not exists:
                logger.info(
                    "Creating Qdrant collection '%s' (dim=%d, distance=Cosine)",
                    self._collection_name, self._embedding_dim,
                )
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self._embedding_dim,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
        except Exception as exc:
            raise VectorDBConnectionError(f"Failed to connect or create Qdrant collection: {exc}") from exc

    def _sync_bm25_from_qdrant(self) -> None:
        try:
            all_chunks: List[Chunk] = []
            offset = None
            while True:
                scroll_res = self._client.scroll(
                    collection_name=self._collection_name,
                    limit=200,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                records, next_offset = scroll_res
                for record in records:
                    payload = record.payload or {}
                    text = payload.get("text", "")
                    if text:
                        cid = payload.get("chunk_id", str(record.id))
                        meta = {k: v for k, v in payload.items() if k != "text"}
                        all_chunks.append(Chunk(chunk_id=cid, text=text, metadata=meta))

                if next_offset is None:
                    break
                offset = next_offset

            if all_chunks:
                self._bm25.add_chunks(all_chunks)
                logger.info("Synced %d chunks from Qdrant into persistent BM25 index", len(all_chunks))
        except Exception as exc:
            logger.warning("Could not sync BM25 index from Qdrant: %s", exc)

    def is_file_ingested(self, file_path: str, content_hash: Optional[str] = None) -> bool:
        """Filter-based deduplication checking either content_hash or normalized file path."""
        try:
            # Check content hash if provided
            if content_hash:
                hash_filter = qmodels.Filter(
                    must=[qmodels.FieldCondition(key="content_hash", match=qmodels.MatchValue(value=content_hash))]
                )
                res, _ = self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=hash_filter,
                    limit=1,
                    with_payload=False,
                )
                if res:
                    return True

            # Check filename / source
            norm_name = Path(file_path).name.lower()
            src_filter = qmodels.Filter(
                should=[
                    qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=file_path)),
                    qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=norm_name)),
                ]
            )
            records, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=src_filter,
                limit=1,
                with_payload=True,
            )
            if records:
                return True

            # Check BM25 chunks
            for c in self._bm25._chunks:
                src = Path(c.metadata.get("source", "")).name.lower()
                if src == norm_name:
                    return True

            return False
        except Exception as exc:
            logger.warning("Error checking if file ingested in Qdrant: %s", exc)
            return False

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        if self._embedding_fn is None:
            raise EmbeddingGenerationError("No embedding function provided to QdrantStoreManager.")

        embeddings = self._embedding_fn(texts)

        # Validate embedding dimensions
        if embeddings and len(embeddings[0]) != self._embedding_dim:
            logger.warning(
                "Embedding dim mismatch: received %d, collection configured for %d. Adjusting.",
                len(embeddings[0]), self._embedding_dim,
            )
            self._embedding_dim = len(embeddings[0])

        points: List[qmodels.PointStruct] = []
        for chunk, emb in zip(chunks, embeddings):
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
    ) -> List[RetrievedChunk]:
        """Hybrid search combining BM25 and Qdrant Dense Vector search using shared RRF."""
        bm25_hits = self._bm25.search(query, top_k=fetch_k)

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

        return reciprocal_rank_fusion(dense_hits=dense_hits, bm25_hits=bm25_hits, top_k=top_k, alpha=alpha)

    def delete_file(self, file_path: str) -> int:
        norm_name = Path(file_path).name.lower()
        try:
            flt = qmodels.Filter(
                should=[
                    qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=file_path)),
                    qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=norm_name)),
                ]
            )
            self._client.delete(collection_name=self._collection_name, points_selector=flt)
            bm25_deleted = self._bm25.remove_file_chunks(file_path)
            logger.info("Deleted chunks for file '%s' from Qdrant and BM25 (%d docs)", norm_name, bm25_deleted)
            return bm25_deleted
        except Exception as exc:
            raise VectorDBConnectionError(f"Failed to delete file '{file_path}' from Qdrant: {exc}") from exc
