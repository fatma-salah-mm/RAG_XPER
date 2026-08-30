"""
rag_xper.core.jobs

Job manager for background asynchronous document ingestion.
Tracks job status, progress percentage, error states, and completion metadata.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


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
    progress: int = 0  # 0 to 100%
    chunks_ingested: int = 0
    strategy_used: str = "recursive"
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class JobManager:
    """Thread-safe in-memory job registry for async background ingestion."""

    def __init__(self) -> None:
        self._jobs: Dict[str, IngestionJob] = {}

    def create_job(self, filename: str, strategy: str = "recursive") -> IngestionJob:
        job_id = str(uuid.uuid4())
        job = IngestionJob(
            job_id=job_id,
            filename=filename,
            strategy_used=strategy,
            status=JobStatus.PENDING,
            progress=0,
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        return self._jobs.get(job_id)

    def update_progress(self, job_id: str, progress: int, status: JobStatus = JobStatus.PROCESSING) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.progress = progress
            job.status = status

    def complete_job(self, job_id: str, chunks_ingested: int) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.chunks_ingested = chunks_ingested
            job.completed_at = time.time()

    def fail_job(self, job_id: str, error_message: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.error = error_message
            job.completed_at = time.time()


job_manager = JobManager()
