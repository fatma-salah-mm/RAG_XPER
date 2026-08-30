"""
tests/test_orchestrator.py for RAG_XPER
"""
import pytest
from unittest.mock import MagicMock
from rag_xper.core.models import Chunk, RetrievedChunk, RAGResponse
from rag_xper.core.generation.rag_orchestrator import RAGOrchestrator


def test_cot_parsing():
    extractor = MagicMock()
    ocr = MagicMock()
    vector_store = MagicMock()
    llm = MagicMock()

    llm.generate.return_value = (
        "Reasoning:\n"
        "1. The document explicitly defines the author as James Clear.\n\n"
        "Answer:\n"
        "The author of the book is James Clear."
    )

    vector_store.hybrid_search.return_value = [
        RetrievedChunk(
            chunk=Chunk(chunk_id="1", text="Atomic Habits by James Clear", metadata={"source": "book.pdf", "page": 1}),
            score=0.95,
        )
    ]

    orch = RAGOrchestrator(extractor=extractor, ocr=ocr, vector_store=vector_store, llm=llm)
    resp = orch.query("Who is the author?")

    assert isinstance(resp, RAGResponse)
    assert "James Clear" in resp.answer
    assert "explicitly defines" in resp.reasoning
    assert len(resp.sources) == 1
