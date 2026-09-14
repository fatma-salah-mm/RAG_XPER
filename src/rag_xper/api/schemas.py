"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(6, ge=1, le=10)
    session_id: str | None = None
    filename: str | None = None
    article_number: str | None = None


class SourceOut(BaseModel):
    source: str | None
    page: int | None
    strategy: str | None
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    reasoning: str | None
    sources: list[SourceOut]
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
    error: str | None = None
    details: dict[str, Any] | None = None


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int


class BookRecord(BaseModel):
    id: int | None = None
    title: str
    author: str | None = None
    category: str | None = "General"
    filename: str
    chunk_count: int | None = 0
    strategy_used: str | None = "recursive"
    status: str | None = "indexed"
    created_at: str | None = None


class FolderIngestRequest(BaseModel):
    directory: str | None = Field(
        None,
        description="Path relative to or inside DOCUMENTS_DIR.",
    )
    strategy: str | None = None
    recursive: bool = False
    force: bool = False
