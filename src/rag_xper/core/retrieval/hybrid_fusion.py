"""
rag_xper.core.retrieval.hybrid_fusion

Unified Reciprocal Rank Fusion (RRF) module combining dense semantic search
and lexical BM25 search.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from rag_xper.core.models import Chunk, RetrievedChunk


def reciprocal_rank_fusion(
    dense_hits: List[Tuple[Chunk, float, int]],  # (chunk, score, rank)
    bm25_hits: List[Tuple[Chunk, float, int]],   # (chunk, score, rank)
    top_k: int = 6,
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> List[RetrievedChunk]:
    """
    Combine dense and lexical hits using weighted Reciprocal Rank Fusion.
    Score = alpha * (1 / (rrf_k + dense_rank + 1)) + (1 - alpha) * (1 / (rrf_k + bm25_rank + 1))
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}

    # Process dense hits
    for chunk, _, rank in dense_hits:
        cid = chunk.chunk_id
        chunk_map[cid] = chunk
        dense_score = alpha * (1.0 / (rrf_k + rank + 1))
        scores[cid] = scores.get(cid, 0.0) + dense_score

    # Process BM25 hits
    for chunk, _, rank in bm25_hits:
        cid = chunk.chunk_id
        chunk_map[cid] = chunk
        bm25_score = (1.0 - alpha) * (1.0 / (rrf_k + rank + 1))
        scores[cid] = scores.get(cid, 0.0) + bm25_score

    # Sort descending by combined RRF score
    sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    results: List[RetrievedChunk] = []
    for cid, rrf_score in sorted_items[:top_k]:
        results.append(RetrievedChunk(chunk=chunk_map[cid], score=float(rrf_score)))

    return results
