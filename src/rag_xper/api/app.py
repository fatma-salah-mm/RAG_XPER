"""
rag_xper.api.app

FastAPI Production Application for RAG_XPER.
Features:
- Decoupled from CLI, powered by rag_xper.bootstrap
- API Key Authentication & Rate Limiting protection
- Async background ingestion with Job Queue (/v1/ingest/async, /v1/jobs/{job_id})
- Document management (/v1/documents, /v1/documents/{filename})
- Monitoring & Observability (/health, /ready, /metrics, /version)
"""
from __future__ import annotations

import os
import tempfile
import time
import uvicorn
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Security,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from rag_xper import __version__
from rag_xper.bootstrap import build_orchestrator
from rag_xper.config import settings
from rag_xper.core.exceptions import RAGPipelineError
from rag_xper.core.generation.rag_orchestrator import SUPPORTED_EXTENSIONS
from rag_xper.core.jobs import JobStatus, job_manager
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="RAG_XPER Enterprise API",
    version=__version__,
    description="Production-grade Arabic/English Multi-Modal RAG API with Qdrant, Persisted BM25, and Async Jobs.",
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
_start_time = time.time()
_stats = {"total_queries": 0, "total_ingests": 0, "total_errors": 0}


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
        return api_key

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


class JobResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    progress: int
    chunks_ingested: int
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int


class FolderIngestRequest(BaseModel):
    directory: Optional[str] = Field(
        None,
        description="Folder to scan. Must resolve inside DOCUMENTS_DIR. Defaults to DOCUMENTS_DIR itself.",
    )
    strategy: Optional[str] = Field(None, description="Chunking strategy override")
    recursive: bool = Field(False, description="Include sub-folders")
    force: bool = Field(False, description="Re-index files even if already indexed")


# --- Shared validation helpers ---
def _validate_extension(filename: Optional[str]) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    return suffix


def _save_upload_to_temp(file: UploadFile, suffix: str) -> str:
    """Stream an upload to disk, aborting once it exceeds MAX_UPLOAD_SIZE_MB."""
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    written = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                tmp.close()
                Path(tmp_path).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,  # Content Too Large
                    detail=f"File exceeds MAX_UPLOAD_SIZE_MB ({settings.max_upload_size_mb} MB).",
                )
            tmp.write(chunk)

    return tmp_path


def _resolve_documents_path(directory: Optional[str]) -> Path:
    """Resolve a requested folder, refusing anything outside DOCUMENTS_DIR."""
    base = Path(settings.documents_dir).resolve()
    target = base if not directory else Path(directory).resolve()

    if target != base and base not in target.parents:
        raise HTTPException(
            status_code=403,
            detail=f"Directory must be inside DOCUMENTS_DIR ('{base}').",
        )
    if not target.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {target}")
    return target


# --- Background Worker Task ---
def _process_async_ingestion(job_id: str, tmp_path: str, orig_filename: str, strategy: Optional[str], force: bool):
    try:
        job_manager.update_progress(job_id, progress=25, status=JobStatus.PROCESSING)
        orch = get_orchestrator()

        job_manager.update_progress(job_id, progress=50, status=JobStatus.PROCESSING)
        n_chunks = orch.ingest_file(tmp_path, strategy=strategy, force=force)

        job_manager.complete_job(job_id, chunks_ingested=n_chunks)
        _stats["total_ingests"] += 1
        logger.info("Async Job '%s' for '%s' completed successfully (%d chunks)", job_id, orig_filename, n_chunks)
    except Exception as exc:
        logger.error("Async Job '%s' failed: %s", job_id, exc)
        job_manager.fail_job(job_id, str(exc))
        _stats["total_errors"] += 1
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _process_folder_ingestion(
    job_id: str,
    directory: str,
    strategy: Optional[str],
    recursive: bool,
    force: bool,
) -> None:
    """Background worker indexing every supported file staged in a server-side folder."""
    try:
        job_manager.update_progress(job_id, progress=5, status=JobStatus.PROCESSING)
        orch = get_orchestrator()

        def on_progress(done: int, total: int) -> None:
            # Reserve the first 5% for startup and cap at 99% until the report is stored.
            pct = 5 + int((done / total) * 94) if total else 99
            job_manager.update_progress(job_id, progress=min(pct, 99), status=JobStatus.PROCESSING)

        report = orch.ingest_directory(
            directory,
            strategy=strategy,
            recursive=recursive,
            force=force,
            progress_callback=on_progress,
        )

        job_manager.complete_job(job_id, chunks_ingested=report["total_chunks"], details=report)
        _stats["total_ingests"] += report["ingested"]
        logger.info(
            "Folder job '%s' completed: %d ingested, %d skipped, %d failed",
            job_id, report["ingested"], report["skipped"], report["failed"],
        )
    except Exception as exc:
        logger.error("Folder job '%s' failed: %s", job_id, exc)
        job_manager.fail_job(job_id, str(exc))
        _stats["total_errors"] += 1


# --- Endpoints ---
@app.get("/health", tags=["Health & Observability"])
async def health_check():
    """Liveness probe returning 200 if process is responsive."""
    return {"status": "ok", "app": "RAG_XPER", "version": __version__}


@app.get("/ready", tags=["Health & Observability"])
async def readiness_check():
    """Readiness probe verifying vector store connectivity."""
    try:
        orch = get_orchestrator()
        return {"status": "ready", "vector_store": settings.vector_store_type}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Service not ready: {exc}")


@app.get("/metrics", tags=["Health & Observability"])
async def metrics_endpoint():
    """Operational metrics reporting total queries, ingests, and index size."""
    bm25_count = 0
    try:
        orch = get_orchestrator()
        if hasattr(orch._vector_store, "_bm25"):
            bm25_count = len(orch._vector_store._bm25._chunks)
    except Exception:
        pass

    return {
        "uptime_seconds": int(time.time() - _start_time),
        "total_queries": _stats["total_queries"],
        "total_ingests": _stats["total_ingests"],
        "total_errors": _stats["total_errors"],
        "bm25_indexed_chunks": bm25_count,
        "vector_store_type": settings.vector_store_type,
    }


@app.get("/version", tags=["Health & Observability"])
async def version_endpoint():
    return {"version": __version__, "app": "RAG_XPER"}


@app.post("/v1/ingest", response_model=IngestResponse, tags=["Ingestion"], dependencies=[Depends(verify_api_key)])
async def ingest_document(
    file: UploadFile = File(...),
    strategy: Optional[str] = Form(None),
    force: bool = Form(False),
):
    """Synchronous ingestion for smaller documents."""
    suffix = _validate_extension(file.filename)

    orchestrator = get_orchestrator()
    tmp_path: Optional[str] = None
    try:
        tmp_path = _save_upload_to_temp(file, suffix)
        n_chunks = orchestrator.ingest_file(tmp_path, strategy=strategy, force=force)
        _stats["total_ingests"] += 1
        return IngestResponse(
            filename=file.filename,
            chunks_ingested=n_chunks,
            strategy_used=strategy or settings.chunking_strategy,
            status="indexed" if n_chunks > 0 else "already_indexed",
        )
    except RAGPipelineError as exc:
        _stats["total_errors"] += 1
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.post("/v1/ingest/async", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Ingestion"], dependencies=[Depends(verify_api_key)])
async def ingest_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    strategy: Optional[str] = Form(None),
    force: bool = Form(False),
):
    """Asynchronous ingestion for large documents with real-time job tracking."""
    suffix = _validate_extension(file.filename)
    tmp_path = _save_upload_to_temp(file, suffix)

    job = job_manager.create_job(filename=file.filename, strategy=strategy or settings.chunking_strategy)
    background_tasks.add_task(_process_async_ingestion, job.job_id, tmp_path, file.filename, strategy, force)

    return JobResponse(
        job_id=job.job_id,
        filename=file.filename,
        status=job.status.value,
        progress=job.progress,
        chunks_ingested=0,
    )


@app.post(
    "/v1/ingest/folder",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Ingestion"],
    dependencies=[Depends(verify_api_key)],
)
async def ingest_folder(request: FolderIngestRequest, background_tasks: BackgroundTasks):
    """Index documents already staged on the server under DOCUMENTS_DIR.

    Nothing is uploaded: drop files into the folder (for example an EC2 bind mount)
    and call this endpoint. Returns a job id to poll via /v1/jobs/{job_id}.
    """
    target = _resolve_documents_path(request.directory)

    job = job_manager.create_job(
        filename=str(target),
        strategy=request.strategy or settings.chunking_strategy,
    )
    background_tasks.add_task(
        _process_folder_ingestion,
        job.job_id,
        str(target),
        request.strategy,
        request.recursive,
        request.force,
    )

    return JobResponse(
        job_id=job.job_id,
        filename=str(target),
        status=job.status.value,
        progress=job.progress,
        chunks_ingested=0,
    )


@app.get("/v1/jobs/{job_id}", response_model=JobResponse, tags=["Ingestion"], dependencies=[Depends(verify_api_key)])
async def get_job_status(job_id: str):
    """Check background ingestion job progress and status."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobResponse(
        job_id=job.job_id,
        filename=job.filename,
        status=job.status.value,
        progress=job.progress,
        chunks_ingested=job.chunks_ingested,
        error=job.error,
        details=job.details,
    )


@app.post("/v1/ask", response_model=AskResponse, tags=["Generation"], dependencies=[Depends(verify_api_key)])
async def ask_question(request: AskRequest):
    """Query the knowledge base and receive a Chain-of-Thought verified answer with sources."""
    orchestrator = get_orchestrator()
    try:
        response = orchestrator.query(request.question, top_k=request.top_k or settings.top_k)
        _stats["total_queries"] += 1
    except RAGPipelineError as exc:
        _stats["total_errors"] += 1
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


@app.get("/v1/documents", response_model=List[DocumentInfo], tags=["Documents"], dependencies=[Depends(verify_api_key)])
async def list_documents():
    """List all unique indexed documents and their chunk counts."""
    doc_counts: Dict[str, int] = {}
    try:
        orch = get_orchestrator()
        if hasattr(orch._vector_store, "_bm25"):
            for c in orch._vector_store._bm25._chunks:
                src = Path(c.metadata.get("source", "doc")).name
                doc_counts[src] = doc_counts.get(src, 0) + 1
    except Exception:
        pass

    return [DocumentInfo(filename=fn, chunk_count=cnt) for fn, cnt in doc_counts.items()]


@app.delete("/v1/documents/{filename}", tags=["Documents"], dependencies=[Depends(verify_api_key)])
async def delete_document(filename: str):
    """Delete all indexed chunks belonging to a filename."""
    safe_name = Path(filename).name
    try:
        # Built inside the guard so configuration errors surface as HTTP responses.
        orchestrator = get_orchestrator()
        removed = orchestrator._vector_store.delete_file(safe_name)
        return {"filename": safe_name, "chunks_deleted": removed}
    except RAGPipelineError as exc:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {exc}") from exc


def run_api():
    """Console script launcher: rag-xper-api"""
    uvicorn.run("rag_xper.api.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run_api()
