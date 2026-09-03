"""
rag_xper.api.app

FastAPI Production Application for RAG_XPER.
Features:
- Decoupled from CLI, powered by rag_xper.bootstrap
- In-Memory & Redis Query Cache Layer for <5ms sub-second responses
- MySQL Database Integration for Books and Chat History (rag_xper_db)
- Modern Enterprise Web Dashboard mounted at /ui
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
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_xper import __version__
from rag_xper.bootstrap import build_orchestrator
from rag_xper.config import settings
from rag_xper.core.cache import query_cache
from rag_xper.core.db.service import delete_book, get_query_history, list_books, log_query, register_book
from rag_xper.core.db.session import init_db
from rag_xper.core.exceptions import RAGPipelineError
from rag_xper.core.generation.rag_orchestrator import SUPPORTED_EXTENSIONS
from rag_xper.core.jobs import JobStatus, job_manager
from rag_xper.core.models import RAGResponse
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="RAG_XPER Enterprise API",
    version=__version__,
    description="Production-grade Arabic/English Multi-Modal RAG API with Qdrant, Persisted BM25, In-Memory Cache, MySQL, and Web UI.",
)

# Initialize Database on Startup
try:
    init_db()
except Exception as db_exc:
    logger.warning("Database startup notice: %s", db_exc)

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
    session_id: Optional[str] = Field(None, description="Optional chat session ID for logging")


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
    is_cached: bool = False
    execution_time_ms: float = 0.0


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


class BookRecord(BaseModel):
    id: Optional[int] = None
    title: str
    author: Optional[str] = None
    category: Optional[str] = "General"
    filename: str
    chunk_count: Optional[int] = 0
    strategy_used: Optional[str] = "recursive"
    status: Optional[str] = "indexed"
    created_at: Optional[str] = None


class FolderIngestRequest(BaseModel):
    directory: Optional[str] = Field(
        None,
        description="Path relative to or inside DOCUMENTS_DIR. Defaults to DOCUMENTS_DIR itself.",
    )
    strategy: Optional[str] = Field(None, description="Chunking strategy to apply to all files in the folder.")
    recursive: bool = Field(False, description="Whether to scan subdirectories.")
    force: bool = Field(False, description="Re-index files even if their content hash is unchanged.")


def _check_supported_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
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
                    status_code=413,
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
def _process_async_ingestion(
    job_id: str,
    tmp_path: str,
    orig_filename: str,
    strategy: Optional[str],
    force: bool,
    title: Optional[str] = None,
    author: Optional[str] = None,
    category: Optional[str] = None,
):
    try:
        job_manager.update_progress(job_id, progress=25, status=JobStatus.PROCESSING)
        orch = get_orchestrator()

        job_manager.update_progress(job_id, progress=50, status=JobStatus.PROCESSING)
        n_chunks = orch.ingest_file(tmp_path, strategy=strategy, force=force)

        # Register in MySQL Books catalog
        register_book(
            title=title or orig_filename,
            author=author,
            category=category or "General",
            filename=orig_filename,
            file_path=tmp_path,
            chunk_count=n_chunks,
            strategy_used=strategy or settings.chunking_strategy,
        )
        query_cache.clear()

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
    try:
        job_manager.update_progress(job_id, progress=5, status=JobStatus.PROCESSING)
        orch = get_orchestrator()

        def on_progress(done: int, total: int) -> None:
            pct = 5 + int((done / total) * 94) if total else 99
            job_manager.update_progress(job_id, progress=min(pct, 99), status=JobStatus.PROCESSING)

        report = orch.ingest_directory(
            directory,
            strategy=strategy,
            recursive=recursive,
            force=force,
            progress_callback=on_progress,
        )

        query_cache.clear()
        job_manager.complete_job(job_id, chunks_ingested=report["total_chunks"], details=report)
        _stats["total_ingests"] += report["ingested"]
        logger.info("Folder job '%s' completed: %d ingested", job_id, report["ingested"])
    except Exception as exc:
        logger.error("Folder job '%s' failed: %s", job_id, exc)
        job_manager.fail_job(job_id, str(exc))
        _stats["total_errors"] += 1


# --- Health & Observability Endpoints ---
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
    """Operational metrics reporting total queries, ingests, and cache statistics."""
    bm25_count = 0
    try:
        orch = get_orchestrator()
        if hasattr(orch._vector_store, "_bm25"):
            bm25_count = len(orch._vector_store._bm25._chunks)
    except Exception:
        pass

    cache_stats = query_cache.get_stats()
    return {
        "uptime_seconds": int(time.time() - _start_time),
        "total_queries": _stats["total_queries"],
        "total_ingests": _stats["total_ingests"],
        "total_errors": _stats["total_errors"],
        "bm25_indexed_chunks": bm25_count,
        "cache_hits": cache_stats["hits"],
        "cache_misses": cache_stats["misses"],
        "cache_size": cache_stats["size"],
        "vector_store_type": settings.vector_store_type,
    }


@app.get("/version", tags=["Health & Observability"])
async def version_endpoint():
    return {"version": __version__, "app": "RAG_XPER"}


# --- Ingestion Endpoints ---
@app.post("/v1/ingest", response_model=IngestResponse, tags=["Ingestion"], dependencies=[Depends(verify_api_key)])
async def ingest_document(
    file: UploadFile = File(...),
    strategy: Optional[str] = Form(None),
    force: bool = Form(False),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
):
    """Synchronous ingestion for smaller documents."""
    suffix = _check_supported_extension(file.filename)
    tmp_path = _save_upload_to_temp(file, suffix)

    orchestrator = get_orchestrator()
    try:
        n_chunks = orchestrator.ingest_file(tmp_path, strategy=strategy, force=force)
        register_book(
            title=title or file.filename,
            author=author,
            category=category or "General",
            filename=file.filename,
            file_path=tmp_path,
            chunk_count=n_chunks,
            strategy_used=strategy or settings.chunking_strategy,
        )
        query_cache.clear()
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
        Path(tmp_path).unlink(missing_ok=True)


@app.post(
    "/v1/ingest/async",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Ingestion"],
    dependencies=[Depends(verify_api_key)],
)
async def ingest_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    strategy: Optional[str] = Form(None),
    force: bool = Form(False),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
):
    """Asynchronous ingestion for large documents with real-time job tracking."""
    suffix = _check_supported_extension(file.filename)
    tmp_path = _save_upload_to_temp(file, suffix)

    job = job_manager.create_job(filename=file.filename, strategy=strategy or settings.chunking_strategy)
    background_tasks.add_task(
        _process_async_ingestion,
        job.job_id,
        tmp_path,
        file.filename,
        strategy,
        force,
        title,
        author,
        category,
    )

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


# --- Generation with Cache Layer ---
@app.post("/v1/ask", response_model=AskResponse, tags=["Generation"], dependencies=[Depends(verify_api_key)])
async def ask_question(request: AskRequest):
    """Query the knowledge base with In-Memory Cache acceleration and MySQL history logging."""
    start_t = time.time()
    effective_top_k = request.top_k or settings.top_k

    # 1. Check In-Memory Query Cache
    if settings.cache_enabled:
        cached_response = query_cache.get(request.question, top_k=effective_top_k)
        if cached_response:
            elapsed_ms = round((time.time() - start_t) * 1000, 2)
            log_query(
                question=request.question,
                answer=cached_response.answer,
                reasoning=cached_response.reasoning,
                sources=[{"source": Path(s.chunk.metadata.get("source", "doc")).name, "score": s.score} for s in cached_response.sources],
                execution_time_ms=elapsed_ms,
                is_cached=True,
                session_id=request.session_id,
            )
            return AskResponse(
                answer=cached_response.answer,
                reasoning=cached_response.reasoning,
                sources=[
                    SourceOut(
                        source=Path(s.chunk.metadata.get("source", "doc")).name,
                        page=s.chunk.metadata.get("page", 1),
                        strategy=s.chunk.metadata.get("strategy", "default"),
                        score=s.score,
                        text=s.chunk.text,
                    )
                    for s in cached_response.sources
                ],
                query=request.question,
                is_cached=True,
                execution_time_ms=elapsed_ms,
            )

    # 2. Query Orchestrator
    orchestrator = get_orchestrator()
    try:
        response: RAGResponse = orchestrator.query(request.question, top_k=effective_top_k)
        _stats["total_queries"] += 1
        elapsed_ms = round((time.time() - start_t) * 1000, 2)

        # Store in Cache
        if settings.cache_enabled:
            query_cache.set(request.question, response, top_k=effective_top_k)

        # Record in MySQL
        log_query(
            question=request.question,
            answer=response.answer,
            reasoning=response.reasoning,
            sources=[{"source": Path(s.chunk.metadata.get("source", "doc")).name, "score": s.score} for s in response.sources],
            execution_time_ms=elapsed_ms,
            is_cached=False,
            session_id=request.session_id,
        )
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
        is_cached=False,
        execution_time_ms=elapsed_ms,
    )


# --- Books & Documents Catalog Endpoints (MySQL) ---
@app.get("/v1/books", response_model=List[BookRecord], tags=["Books (MySQL)"], dependencies=[Depends(verify_api_key)])
async def get_books_catalog(category: Optional[str] = None):
    """Retrieve catalog of books and documents from MySQL database."""
    books = list_books(category=category)
    return [BookRecord(**b) for b in books]


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
    """Delete all indexed chunks belonging to a filename from Qdrant, BM25, and MySQL."""
    safe_name = Path(filename).name
    try:
        orchestrator = get_orchestrator()
        removed = orchestrator._vector_store.delete_file(safe_name)
        delete_book(safe_name)
        query_cache.clear()
        return {"filename": safe_name, "chunks_deleted": removed}
    except RAGPipelineError as exc:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {exc}") from exc


# --- Mount Web UI Dashboard ---
ui_dir = Path(__file__).resolve().parent.parent.parent.parent / "apps" / "web_dashboard"
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/ui/")


def run_api():
    """Console script launcher: rag-xper-api"""
    uvicorn.run("rag_xper.api.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run_api()
