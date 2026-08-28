"""
api.py for RAG_XPER

FastAPI REST backend exposing RAG_XPER endpoints.
Run with:
    uvicorn api:app --reload --port 8000
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from core.exceptions import RAGPipelineError
from main import build_orchestrator
from utils.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="RAG_XPER Enterprise API", version="2.0.0")

_orchestrator = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = build_orchestrator()
    return _orchestrator


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 6


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


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    strategy: Optional[str] = Form(None),
):
    orchestrator = get_orchestrator()
    suffix = Path(file.filename).suffix
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        n_chunks = orchestrator.ingest_file(tmp_path, strategy=strategy)
        return {
            "filename": file.filename,
            "chunks_ingested": n_chunks,
            "strategy_used": strategy or "default",
        }
    except RAGPipelineError as exc:
        logger.error("Ingestion failed for '%s': %s", file.filename, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    orchestrator = get_orchestrator()
    try:
        response = orchestrator.query(request.question, top_k=request.top_k or 6)
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
    )


@app.get("/health")
async def health():
    return {"status": "ok", "app": "RAG_XPER"}
