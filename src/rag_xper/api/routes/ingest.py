"""Document ingestion endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status

from rag_xper.api import state
from rag_xper.api.dependencies import verify_api_key
from rag_xper.api.helpers import check_supported_extension, resolve_documents_path, save_upload_to_temp
from rag_xper.api.schemas import FolderIngestRequest, IngestResponse, JobResponse
from rag_xper.api.workers import process_async_ingestion, process_folder_ingestion
from rag_xper.config import settings
from rag_xper.core.cache import query_cache
from rag_xper.core.db.service import register_book
from rag_xper.core.exceptions import RAGPipelineError
from rag_xper.core.jobs import job_manager

router = APIRouter(prefix="/v1", tags=["Ingestion"], dependencies=[Depends(verify_api_key)])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    strategy: str | None = Form(None),
    force: bool = Form(False),
    title: str | None = Form(None),
    author: str | None = Form(None),
    category: str | None = Form(None),
):
    suffix = check_supported_extension(file.filename)
    tmp_path = save_upload_to_temp(file, suffix)

    orchestrator = state.get_orchestrator()
    stats = state.get_stats()
    try:
        n_chunks = orchestrator.ingest_file(
            tmp_path,
            strategy=strategy,
            force=force,
            original_filename=file.filename,
        )
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
        stats["total_ingests"] += 1
        return IngestResponse(
            filename=file.filename,
            chunks_ingested=n_chunks,
            strategy_used=strategy or settings.chunking_strategy,
            status="indexed" if n_chunks > 0 else "already_indexed",
        )
    except RAGPipelineError as exc:
        stats["total_errors"] += 1
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/ingest/async", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    strategy: str | None = Form(None),
    force: bool = Form(False),
    title: str | None = Form(None),
    author: str | None = Form(None),
    category: str | None = Form(None),
):
    suffix = check_supported_extension(file.filename)
    tmp_path = save_upload_to_temp(file, suffix)

    job = job_manager.create_job(filename=file.filename, strategy=strategy or settings.chunking_strategy)
    background_tasks.add_task(
        process_async_ingestion,
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


@router.post("/ingest/folder", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_folder(request: FolderIngestRequest, background_tasks: BackgroundTasks):
    target = resolve_documents_path(request.directory)

    job = job_manager.create_job(
        filename=str(target),
        strategy=request.strategy or settings.chunking_strategy,
    )
    background_tasks.add_task(
        process_folder_ingestion,
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


@router.get("/jobs/{job_id}", response_model=JobResponse)
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
