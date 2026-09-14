"""Question-answering endpoint."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from rag_xper.api import state
from rag_xper.api.dependencies import verify_api_key
from rag_xper.api.rate_limit import _ask_rate_limit, limiter
from rag_xper.api.schemas import AskRequest, AskResponse, SourceOut
from rag_xper.config import settings
from rag_xper.core.cache import query_cache
from rag_xper.core.db.service import log_query
from rag_xper.core.exceptions import RAGPipelineError
from rag_xper.core.models import RAGResponse
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Generation"], dependencies=[Depends(verify_api_key)])


def _sources_payload(response: RAGResponse) -> list[SourceOut]:
    return [
        SourceOut(
            source=Path(s.chunk.metadata.get("source", "doc")).name,
            page=s.chunk.metadata.get("page", 1),
            strategy=s.chunk.metadata.get("strategy", "default"),
            score=s.score,
            text=s.chunk.text,
        )
        for s in response.sources
    ]


@router.post("/ask", response_model=AskResponse)
@limiter.limit(_ask_rate_limit())
async def ask_question(request: Request, payload: AskRequest):
    start_t = time.time()
    effective_top_k = payload.top_k or settings.top_k
    stats = state.get_stats()

    if settings.cache_enabled:
        cached_response = query_cache.get(payload.question, top_k=effective_top_k)
        if cached_response:
            elapsed_ms = round((time.time() - start_t) * 1000, 2)
            log_query(
                question=payload.question,
                answer=cached_response.answer,
                reasoning=cached_response.reasoning,
                sources=[
                    {"source": Path(s.chunk.metadata.get("source", "doc")).name, "score": s.score}
                    for s in cached_response.sources
                ],
                execution_time_ms=elapsed_ms,
                is_cached=True,
                session_id=payload.session_id,
            )
            return AskResponse(
                answer=cached_response.answer,
                reasoning=cached_response.reasoning,
                sources=_sources_payload(cached_response),
                query=payload.question,
                is_cached=True,
                execution_time_ms=elapsed_ms,
            )

    orchestrator = state.get_orchestrator()
    try:
        response: RAGResponse = orchestrator.query(
            payload.question,
            top_k=effective_top_k,
            filter_filename=payload.filename,
            filter_article=payload.article_number,
        )
        stats["total_queries"] += 1
        elapsed_ms = round((time.time() - start_t) * 1000, 2)

        if settings.cache_enabled:
            query_cache.set(payload.question, response, top_k=effective_top_k)

        log_query(
            question=payload.question,
            answer=response.answer,
            reasoning=response.reasoning,
            sources=[
                {"source": Path(s.chunk.metadata.get("source", "doc")).name, "score": s.score} for s in response.sources
            ],
            execution_time_ms=elapsed_ms,
            is_cached=False,
            session_id=payload.session_id,
        )
        logger.info(
            "ask completed top_k=%d n_hits=%d latency_ms=%.1f cached=false",
            effective_top_k,
            len(response.sources),
            elapsed_ms,
        )
    except RAGPipelineError as exc:
        stats["total_errors"] += 1
        logger.error("Query failed for '%s': %s", payload.question[:80], exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AskResponse(
        answer=response.answer,
        reasoning=response.reasoning,
        sources=_sources_payload(response),
        query=payload.question,
        is_cached=False,
        execution_time_ms=elapsed_ms,
    )
