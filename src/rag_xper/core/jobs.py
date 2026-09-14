"""
rag_xper.core.jobs

Job manager for background asynchronous document ingestion.
Supports in-memory (single replica) and Redis (multi-replica) backends.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from rag_xper.config import settings
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)

_JOB_KEY_PREFIX = "rag_xper:job:"
_JOB_TTL_SECONDS = 7 * 24 * 3600


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestionJob:
    job_id: str
    filename: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    chunks_ingested: int = 0
    strategy_used: str = "recursive"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    details: dict[str, Any] | None = None


class JobManagerBackend(Protocol):
    def create_job(self, filename: str, strategy: str = "recursive") -> IngestionJob: ...

    def get_job(self, job_id: str) -> IngestionJob | None: ...

    def update_progress(self, job_id: str, progress: int, status: JobStatus = JobStatus.PROCESSING) -> None: ...

    def complete_job(
        self,
        job_id: str,
        chunks_ingested: int,
        details: dict[str, Any] | None = None,
    ) -> None: ...

    def fail_job(self, job_id: str, error_message: str) -> None: ...


def _job_to_dict(job: IngestionJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "filename": job.filename,
        "status": job.status.value,
        "progress": job.progress,
        "chunks_ingested": job.chunks_ingested,
        "strategy_used": job.strategy_used,
        "error": job.error,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "details": job.details,
    }


def _job_from_dict(data: dict[str, Any]) -> IngestionJob:
    return IngestionJob(
        job_id=data["job_id"],
        filename=data["filename"],
        status=JobStatus(data["status"]),
        progress=int(data.get("progress", 0)),
        chunks_ingested=int(data.get("chunks_ingested", 0)),
        strategy_used=data.get("strategy_used", "recursive"),
        error=data.get("error"),
        created_at=float(data.get("created_at", time.time())),
        completed_at=data.get("completed_at"),
        details=data.get("details"),
    )


class MemoryJobManager:
    """Thread-safe in-memory job registry for async background ingestion."""

    def __init__(self) -> None:
        self._jobs: dict[str, IngestionJob] = {}
        self._lock = threading.Lock()

    def create_job(self, filename: str, strategy: str = "recursive") -> IngestionJob:
        job_id = str(uuid.uuid4())
        job = IngestionJob(
            job_id=job_id,
            filename=filename,
            strategy_used=strategy,
            status=JobStatus.PENDING,
            progress=0,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> IngestionJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update_progress(self, job_id: str, progress: int, status: JobStatus = JobStatus.PROCESSING) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.progress = progress
                job.status = status

    def complete_job(
        self,
        job_id: str,
        chunks_ingested: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.COMPLETED
                job.progress = 100
                job.chunks_ingested = chunks_ingested
                job.details = details
                job.completed_at = time.time()

    def fail_job(self, job_id: str, error_message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error = error_message
                job.completed_at = time.time()


class RedisJobManager:
    """Redis-backed job registry for multi-replica API deployments."""

    def __init__(self, redis_url: str) -> None:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("redis package is required when REDIS_URL is set. Install with: uv sync --extra redis") from exc

        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def _key(self, job_id: str) -> str:
        return f"{_JOB_KEY_PREFIX}{job_id}"

    def _save(self, job: IngestionJob) -> None:
        self._client.setex(self._key(job.job_id), _JOB_TTL_SECONDS, json.dumps(_job_to_dict(job)))

    def create_job(self, filename: str, strategy: str = "recursive") -> IngestionJob:
        job_id = str(uuid.uuid4())
        job = IngestionJob(
            job_id=job_id,
            filename=filename,
            strategy_used=strategy,
            status=JobStatus.PENDING,
            progress=0,
        )
        self._save(job)
        return job

    def get_job(self, job_id: str) -> IngestionJob | None:
        raw = self._client.get(self._key(job_id))
        if not raw:
            return None
        return _job_from_dict(json.loads(raw))

    def update_progress(self, job_id: str, progress: int, status: JobStatus = JobStatus.PROCESSING) -> None:
        job = self.get_job(job_id)
        if job:
            job.progress = progress
            job.status = status
            self._save(job)

    def complete_job(
        self,
        job_id: str,
        chunks_ingested: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        job = self.get_job(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.chunks_ingested = chunks_ingested
            job.details = details
            job.completed_at = time.time()
            self._save(job)

    def fail_job(self, job_id: str, error_message: str) -> None:
        job = self.get_job(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.error = error_message
            job.completed_at = time.time()
            self._save(job)


_manager: JobManagerBackend | None = None
_manager_lock = threading.Lock()


def get_job_manager() -> JobManagerBackend:
    """Return the configured job backend (Redis when REDIS_URL is set, else in-memory)."""
    global _manager
    with _manager_lock:
        if _manager is None:
            if settings.redis_url:
                logger.info("Using Redis job backend: %s", settings.redis_url.split("@")[-1])
                _manager = RedisJobManager(settings.redis_url)
            else:
                _manager = MemoryJobManager()
        return _manager


class _JobManagerProxy:
    """Lazy proxy so imports keep working: `from rag_xper.core.jobs import job_manager`."""

    def create_job(self, filename: str, strategy: str = "recursive") -> IngestionJob:
        return get_job_manager().create_job(filename, strategy)

    def get_job(self, job_id: str) -> IngestionJob | None:
        return get_job_manager().get_job(job_id)

    def update_progress(self, job_id: str, progress: int, status: JobStatus = JobStatus.PROCESSING) -> None:
        get_job_manager().update_progress(job_id, progress, status)

    def complete_job(
        self,
        job_id: str,
        chunks_ingested: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        get_job_manager().complete_job(job_id, chunks_ingested, details)

    def fail_job(self, job_id: str, error_message: str) -> None:
        get_job_manager().fail_job(job_id, error_message)


job_manager = _JobManagerProxy()
