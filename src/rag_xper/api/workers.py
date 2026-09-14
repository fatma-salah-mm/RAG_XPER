"""Background ingestion workers for async API jobs."""

from __future__ import annotations

from pathlib import Path

from rag_xper.api import state
from rag_xper.config import settings
from rag_xper.core.cache import query_cache
from rag_xper.core.db.service import register_book
from rag_xper.core.jobs import JobStatus, job_manager
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)


def process_async_ingestion(
    job_id: str,
    tmp_path: str,
    orig_filename: str,
    strategy: str | None,
    force: bool,
    title: str | None = None,
    author: str | None = None,
    category: str | None = None,
) -> None:
    stats = state.get_stats()
    try:
        job_manager.update_progress(job_id, progress=25, status=JobStatus.PROCESSING)
        orch = state.get_orchestrator()

        job_manager.update_progress(job_id, progress=50, status=JobStatus.PROCESSING)
        n_chunks = orch.ingest_file(tmp_path, strategy=strategy, force=force, original_filename=orig_filename)

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
        stats["total_ingests"] += 1
        logger.info("Async Job '%s' for '%s' completed successfully (%d chunks)", job_id, orig_filename, n_chunks)
    except Exception as exc:
        logger.error("Async Job '%s' failed: %s", job_id, exc)
        job_manager.fail_job(job_id, str(exc))
        stats["total_errors"] += 1
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def process_folder_ingestion(
    job_id: str,
    directory: str,
    strategy: str | None,
    recursive: bool,
    force: bool,
) -> None:
    stats = state.get_stats()
    try:
        job_manager.update_progress(job_id, progress=5, status=JobStatus.PROCESSING)
        orch = state.get_orchestrator()

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
        stats["total_ingests"] += report["ingested"]
        logger.info("Folder job '%s' completed: %d ingested", job_id, report["ingested"])
    except Exception as exc:
        logger.error("Folder job '%s' failed: %s", job_id, exc)
        job_manager.fail_job(job_id, str(exc))
        stats["total_errors"] += 1
