"""Document catalog and deletion endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from rag_xper.api import state
from rag_xper.api.dependencies import verify_api_key
from rag_xper.api.schemas import BookRecord, DocumentInfo
from rag_xper.core.cache import query_cache
from rag_xper.core.db.service import delete_book, list_books
from rag_xper.core.exceptions import RAGPipelineError

router = APIRouter(prefix="/v1", tags=["Documents"])


@router.get("/books", response_model=list[BookRecord], dependencies=[Depends(verify_api_key)])
async def get_books_catalog(category: str | None = None):
    books = list_books(category=category)
    return [BookRecord(**b) for b in books]


@router.get("/documents", response_model=list[DocumentInfo], dependencies=[Depends(verify_api_key)])
async def list_documents():
    doc_counts: dict[str, int] = {}
    try:
        orch = state.get_orchestrator()
        if hasattr(orch._vector_store, "get_indexed_documents"):
            doc_counts = orch._vector_store.get_indexed_documents()
        elif hasattr(orch._vector_store, "_bm25"):
            for c in orch._vector_store._bm25._chunks:
                src = c.metadata.get("filename") or Path(c.metadata.get("source", "doc")).name
                doc_counts[src] = doc_counts.get(src, 0) + 1
    except Exception:
        pass

    return [DocumentInfo(filename=fn, chunk_count=cnt) for fn, cnt in sorted(doc_counts.items())]


@router.delete("/documents/{filename}", dependencies=[Depends(verify_api_key)])
async def delete_document(filename: str):
    safe_name = Path(filename).name
    try:
        orchestrator = state.get_orchestrator()
        removed = orchestrator._vector_store.delete_file(safe_name)
        delete_book(safe_name)
        query_cache.clear()
        return {"filename": safe_name, "chunks_deleted": removed}
    except RAGPipelineError as exc:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {exc}") from exc
