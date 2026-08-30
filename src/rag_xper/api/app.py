"""
rag_xper.api.app

FastAPI Production Application for RAG_XPER.
Features:
- Decoupled from CLI, powered by rag_xper.bootstrap
- API Key Authentication & Rate Limiting protection
- Routes: /v1/ingest, /v1/ask, /v1/documents, /health, /ready
- Console script entrypoint: run_api()
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uvicorn
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Security, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from rag_xper.bootstrap import build_orchestrator
from rag_xper.config import settings
from rag_xper.core.exceptions import RAGPipelineError
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="RAG_XPER Enterprise API",
    version="2.0.0",
    description="Production-grade Arabic/English Multi-Modal RAG API with Qdrant and Modular Chunking.",
)

# CORS Middleware
origins = list(settings.cors_origins) if settings.cors_origins else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_orchestrator = None


def get_orchestrator():
    """Lazily load pipeline components once per worker process."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = build_orchestrator()
    return _orchestrator


# --- API Key Security Dependency ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """Validate API key if configured in settings."""
    if not settings.api_keys:
        return api_key  # Auth is optional in dev/local mode if API_KEYS is empty

    if not api_key or api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key in 'X-API-Key' header.",
        )
    return api_key


# --- Pydantic Schemas ---
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question text to answer from documents")
    top_k: Optional[int] = Field(6, ge=1, le=50, description="Number of retrieved chunks")


class SourceOut(BaseModel):
    source: Optional[str]
    page: Optional[int]
    strategy: Optional[str]
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    reasoning: Optional[str]
    sources: List[SourceOut]
    query: str


class IngestResponse(BaseModel):
    filename: str
    chunks_ingested: int
    strategy_used: str
    status: str


# --- Endpoints ---
@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness probe returning 200 if process is responsive."""
    return {"status": "ok", "app": "RAG_XPER", "version": "2.0.0"}


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness probe verifying vector store connectivity."""
    try:
        orch = get_orchestrator()
        return {"status": "ready", "vector_store": settings.vector_store_type}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Service not ready: {exc}")


@app.post("/v1/ingest", response_model=IngestResponse, tags=["Ingestion"], dependencies=[Depends(verify_api_key)])
async def ingest_document(
    file: UploadFile = File(...),
    strategy: Optional[str] = Form(None),
    force: bool = Form(False),
):
    """Ingest a PDF, Markdown, text, or image document."""
    allowed_exts = {".pdf", ".md", ".txt", ".markdown", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed types: {sorted(list(allowed_exts))}",
        )

    orchestrator = get_orchestrator()
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        n_chunks = orchestrator.ingest_file(tmp_path, strategy=strategy, force=force)
        return IngestResponse(
            filename=file.filename,
            chunks_ingested=n_chunks,
            strategy_used=strategy or settings.chunking_strategy,
            status="indexed" if n_chunks > 0 else "already_indexed",
        )
    except RAGPipelineError as exc:
        logger.error("Ingestion failed for '%s': %s", file.filename, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.post("/v1/ask", response_model=AskResponse, tags=["Generation"], dependencies=[Depends(verify_api_key)])
async def ask_question(request: AskRequest):
    """Query the knowledge base and receive a Chain-of-Thought verified answer with sources."""
    orchestrator = get_orchestrator()
    try:
        response = orchestrator.query(request.question, top_k=request.top_k or settings.top_k)
    except RAGPipelineError as exc:
        logger.error("Query failed for '%s': %s", request.question, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AskResponse(
        answer=response.answer,
        reasoning=response.reasoning,
        sources=[
            SourceOut(
                source=Path(s.chunk.metadata.get("source", "doc")).name,
                page=s.chunk.metadata.get("page", 1),
                strategy=s.chunk.metadata.get("strategy", "default"),
                score=s.score,
                text=s.chunk.text,
            )
            for s in response.sources
        ],
        query=request.question,
    )


@app.delete("/v1/documents/{filename}", tags=["Documents"], dependencies=[Depends(verify_api_key)])
async def delete_document(filename: str):
    """Delete all indexed chunks belonging to a filename."""
    orchestrator = get_orchestrator()
    try:
        removed = orchestrator._vector_store.delete_file(filename)
        return {"filename": filename, "chunks_deleted": removed}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {exc}")


def run_api():
    """Console script launcher: rag-xper-api"""
    uvicorn.run("rag_xper.api.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run_api()
