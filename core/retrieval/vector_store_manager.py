"""
core/retrieval/vector_store_manager.py for RAG_XPER

Vector store manager supporting ChromaDB as an alternative store,
plus VectorStoreFactory for transparent instantiation between Qdrant and ChromaDB.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.exceptions import EmbeddingGenerationError, VectorDBConnectionError
from core.models import Chunk, RetrievedChunk
from core.retrieval.base_vector_store import BaseVectorStore
from core.retrieval.bm25_retriever import BM25Retriever
from core.retrieval.qdrant_store_manager import QdrantStoreManager
from utils.logger import get_logger

logger = get_logger(__name__)


class ChromaVectorStoreManager(BaseVectorStore):
    """ChromaDB Vector Store implementation with integrated BM25 Hybrid Search."""

    def __init__(
        self,
        db_path: str = "./storage/chroma_db",
        collection_name: str = "rag_xper_documents",
        embedding_dim: int = 3072,
        embedding_fn=None,
    ) -> None:
        self._db_path = db_path
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim
        self._embedding_fn = embedding_fn
        self._bm25 = BM25Retriever()

        Path(db_path).mkdir(parents=True, exist_ok=True)

        try:
            self._client = chromadb.PersistentClient(
                path=db_path,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._sync_bm25_from_chroma()
            logger.info(
                "ChromaVectorStoreManager ready (collection='%s', bm25_docs=%d)",
                collection_name,
                self._bm25.total_docs,
            )
        except Exception as exc:
            raise VectorDBConnectionError(
                f"Could not initialise ChromaDB at '{db_path}': {exc}"
            ) from exc

    def _sync_bm25_from_chroma(self) -> None:
        try:
            self._bm25.clear()
            all_data = self._collection.get(include=["documents", "metadatas"])
            synced_chunks: List[Chunk] = []
            if all_data and all_data.get("ids"):
                for cid, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
                    synced_chunks.append(Chunk(chunk_id=cid, text=doc, metadata=meta or {}))
            if synced_chunks:
                self._bm25.add_chunks(synced_chunks)
                logger.info("Synced %d chunks from ChromaDB into BM25 index", len(synced_chunks))
        except Exception as exc:
            logger.warning("Could not sync BM25 index from ChromaDB: %s", exc)

    def is_file_ingested(self, file_path: str) -> bool:
        try:
            target_norm = Path(file_path).name
            all_meta = self._collection.get(include=["metadatas"])
            if all_meta and all_meta.get("metadatas"):
                for m in all_meta["metadatas"]:
                    src = m.get("source", "")
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
            raise EmbeddingGenerationError("No embedding function provided.")

        try:
            embeddings = self._embedding_fn(texts)
        except Exception as exc:
            raise EmbeddingGenerationError(f"Embedding generation failed: {exc}") from exc

        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Ingest in slices of 64
        batch_size = 64
        for idx in range(0, len(chunks), batch_size):
            b_ids = ids[idx : idx + batch_size]
            b_emb = embeddings[idx : idx + batch_size]
            b_docs = texts[idx : idx + batch_size]
            b_meta = metadatas[idx : idx + batch_size]
            self._collection.upsert(
                ids=b_ids,
                embeddings=b_emb,
                documents=b_docs,
                metadatas=b_meta,
            )

        self._bm25.add_chunks(chunks)
        return len(chunks)

    def similarity_search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        if self._embedding_fn is None:
            raise EmbeddingGenerationError("No embedding function provided.")

        try:
            query_vector = self._embedding_fn([query])[0]
        except Exception as exc:
            logger.warning("Dense embedding failed (%s). Falling back to BM25.", exc)
            bm25_results = self._bm25.search(query, top_k=top_k)
            return [RetrievedChunk(chunk=c, score=s) for c, s, _ in bm25_results]

        raw = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        results: List[RetrievedChunk] = []
        if raw["ids"] and raw["ids"][0]:
            for cid, doc, meta, dist in zip(
                raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
            ):
                similarity = max(0.0, 1.0 - (dist or 0.0))
                results.append(
                    RetrievedChunk(
                        chunk=Chunk(chunk_id=cid, text=doc, metadata=meta or {}),
                        score=round(similarity, 4),
                    )
                )
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 6,
        fetch_k: int = 25,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> List[RetrievedChunk]:
        bm25_hits = self._bm25.search(query, top_k=fetch_k)

        dense_hits = []
        try:
            query_vector = self._embedding_fn([query])[0]
            raw = self._collection.query(
                query_embeddings=[query_vector],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances"],
            )
            if raw["ids"] and raw["ids"][0]:
                for dense_rank, (cid, doc, meta, dist) in enumerate(
                    zip(raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0])
                ):
                    similarity = max(0.0, 1.0 - (dist or 0.0))
                    dense_hits.append(
                        (Chunk(chunk_id=cid, text=doc, metadata=meta or {}), similarity, dense_rank)
                    )
        except Exception as exc:
            logger.warning("Dense search failed in hybrid search (%s). Returning BM25.", exc)
            return [RetrievedChunk(chunk=c, score=s) for c, s, _ in bm25_hits[:top_k]]

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
            data = self._collection.get(include=["metadatas"])
            ids_to_del = []
            for cid, meta in zip(data["ids"], data["metadatas"]):
                src = (meta or {}).get("source", "")
                if target_name in src or Path(src).name == target_name:
                    ids_to_del.append(cid)

            if ids_to_del:
                self._collection.delete(ids=ids_to_del)
                self._sync_bm25_from_chroma()
            return len(ids_to_del)
        except Exception as exc:
            logger.error("Failed to delete file '%s' from ChromaDB: %s", file_path, exc)
            return 0


class VectorStoreFactory:
    """Factory creating either Qdrant or ChromaDB vector store based on configuration."""

    @staticmethod
    def create_vector_store(config, embedding_fn) -> BaseVectorStore:
        store_type = getattr(config, "vector_store_type", "qdrant").lower()

        if store_type == "qdrant":
            logger.info("Initializing Qdrant Vector Store at '%s'", config.qdrant_storage_path)
            return QdrantStoreManager(
                storage_path=config.qdrant_storage_path,
                collection_name=config.collection_name,
                embedding_fn=embedding_fn,
            )
        else:
            logger.info("Initializing ChromaDB Vector Store at '%s'", config.vector_db_path)
            return ChromaVectorStoreManager(
                db_path=config.vector_db_path,
                collection_name=config.collection_name,
                embedding_fn=embedding_fn,
            )
