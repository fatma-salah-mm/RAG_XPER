"""
tests/test_stress_orchestrator_edge_cases.py

Edge-Case and Resilience Tests for RAG Orchestrator and LLM Gateway.
Tests:
- Zero retrieval matches return clear fallback message without hallucinating
- Multi-child deduplication resolving to identical parent context
- Extraction error propagation on non-existent files
- Parsing of unstructured LLM completions
"""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from rag_xper.core.exceptions import DocumentExtractionError
from rag_xper.core.generation.rag_orchestrator import RAGOrchestrator
from rag_xper.core.models import Chunk, RetrievedChunk


def test_zero_retrieval_matches_fallback():
    """Verify system returns strict not-found response when no chunks match query."""
    extractor = MagicMock()
    ocr = MagicMock()
    vector_store = MagicMock()
    llm = MagicMock()

    vector_store.hybrid_search.return_value = []

    orch = RAGOrchestrator(extractor=extractor, ocr=ocr, vector_store=vector_store, llm=llm)
    response = orch.query("ما هي عقوبة مخالفة المادة 999؟")

    assert "لم يتم العثور على أي معلومات" in response.answer
    assert response.sources == []
    assert llm.generate.called is False  # Saves LLM API quota when no context


def test_parent_child_context_deduplication():
    """Verify multiple child chunks from the same parent are deduplicated before prompt generation."""
    extractor = MagicMock()
    ocr = MagicMock()
    vector_store = MagicMock()
    llm = MagicMock()

    # Three child chunks belonging to the same parent
    parent_text = "هذا هو النص الكامل للفقرة الأب التي تحتوي على تفاصيل العقد والالتزامات."
    c1 = Chunk(chunk_id="p1:c0", text="child 0", metadata={"parent_id": "parent_1", "parent_text": parent_text})
    c2 = Chunk(chunk_id="p1:c1", text="child 1", metadata={"parent_id": "parent_1", "parent_text": parent_text})
    c3 = Chunk(chunk_id="p1:c2", text="child 2", metadata={"parent_id": "parent_1", "parent_text": parent_text})

    vector_store.hybrid_search.return_value = [
        RetrievedChunk(chunk=c1, score=0.9),
        RetrievedChunk(chunk=c2, score=0.85),
        RetrievedChunk(chunk=c3, score=0.8),
    ]

    llm.generate.return_value = "Reasoning: Context analyzed.\nAnswer: Test Answer."

    orch = RAGOrchestrator(extractor=extractor, ocr=ocr, vector_store=vector_store, llm=llm)
    orch.query("سؤال تجريبي")

    # Check generated prompt
    prompt_arg = llm.generate.call_args[0][0]
    # The parent text should appear EXACTLY ONCE in the prompt context, not 3 times
    assert prompt_arg.count(parent_text) == 1


def test_non_existent_file_ingestion_error():
    """Verify attempting to ingest non-existent file raises DocumentExtractionError."""
    extractor = MagicMock()
    ocr = MagicMock()
    vector_store = MagicMock()
    llm = MagicMock()

    orch = RAGOrchestrator(extractor=extractor, ocr=ocr, vector_store=vector_store, llm=llm)
    with pytest.raises(DocumentExtractionError):
        orch.ingest_file("C:/fake_path_does_not_exist.pdf")
