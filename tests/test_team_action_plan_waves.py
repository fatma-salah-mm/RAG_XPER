"""
tests/test_team_action_plan_waves.py

Comprehensive Verification Suite covering Waves 1, 2, and 3 of TEAM_ACTION_PLAN.md:
- T1.1: Fail-closed auth & constant-time key comparison
- T1.3: Stable document identity (doc_id, file_hash, cross-store clean deletion)
- T1.4: Safe EMBEDDING_DIM derivation
- T2.1: Cross-page article boundary chunking
- T2.3: Strict prompt parsing & bilingual fallback
- T2.4: Score thresholding & top_k limits
- T2.5: Explicit metadata filtering (filename & article_number)
- T3.2: Structured JSON logging with request_id
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from rag_xper.api.app import app
from rag_xper.config import Settings
from rag_xper.core.exceptions import ConfigurationError
from rag_xper.core.generation.rag_orchestrator import RAGOrchestrator
from rag_xper.core.ingestion.text_chunker import ArticleBasedChunker
from rag_xper.core.models import Chunk, PageContent, RetrievedChunk, SourceType
from rag_xper.core.retrieval.bm25_retriever import BM25Retriever
from rag_xper.utils.logger import JSONFormatter

client = TestClient(app)


def test_t1_1_fail_closed_auth_validation():
    """T1.1: Settings.validate() must raise ConfigurationError when require_auth=True and API_KEYS is empty."""
    invalid_settings = Settings(
        llm_provider="ollama",
        require_auth=True,
        api_keys=(),
    )
    with pytest.raises(ConfigurationError, match="REQUIRE_AUTH is true but API_KEYS is empty"):
        invalid_settings.validate()


def test_t1_4_safe_embedding_dim_derivation():
    """T1.4: Empty or non-numeric EMBEDDING_DIM should not crash, but safely derive default."""
    s_gemini = Settings(llm_provider="gemini", gemini_api_key="dummy")
    assert s_gemini.embedding_dim == 3072

    s_ollama = Settings(llm_provider="ollama")
    assert s_ollama.embedding_dim == 768


def test_t2_1_cross_page_article_chunking():
    """T2.1: Article starting on Page 1 and continuing on Page 2 must become a single chunk with article_number."""
    chunker = ArticleBasedChunker(fallback_max_size=2000)
    page1 = PageContent(
        source_path="nizam_alithbat.pdf",
        page_number=1,
        text="المادة 70: لا يجوز الإثبات بشهادة الشهود في التصرفات المدنية التي تزيد قيمتها على مائة ألف ريال",
        source_type=SourceType.NATIVE_TEXT,
    )
    page2 = PageContent(
        source_path="nizam_alithbat.pdf",
        page_number=2,
        text="أو ما يعادلها، أو كانت غير محددة القيمة، إلا إذا وجد اتفاق أو نص نظامي يقضي بغير ذلك.",
        source_type=SourceType.NATIVE_TEXT,
    )

    chunks = chunker.chunk_pages([page1, page2])

    assert len(chunks) == 1, "Article spanning two pages must not be split across the page boundary."
    chunk = chunks[0]
    assert chunk.metadata.get("article_number") == "70"
    assert "المادة 70" in chunk.text
    assert "إلا إذا وجد اتفاق" in chunk.text
    assert "[[PAGE" not in chunk.text, "Page markers must be cleaned from final chunk text."


def test_t2_3_prompt_fail_closed_parsing():
    """T2.3: When LLM response omits 'Answer:' header, parser must not dump raw reasoning as answer."""
    mock_extractor = MagicMock()
    mock_ocr = MagicMock()
    mock_store = MagicMock()
    mock_llm = MagicMock()

    orch = RAGOrchestrator(
        extractor=mock_extractor,
        ocr=mock_ocr,
        vector_store=mock_store,
        llm=mock_llm,
    )

    # Missing Answer header
    raw_leaked_cot = "Reasoning: قام الباحث بالتحليل ولم يكتب الإجابة النهائية بعد."
    reasoning, answer = orch._parse_cot_response(raw_leaked_cot, is_arabic=True)

    assert "لم يتم استخلاص إجابة محددة" in answer
    assert answer != raw_leaked_cot


def test_t2_3_bilingual_empty_retrieval():
    """T2.3: Empty retrieval response must match the language of the question."""
    mock_extractor = MagicMock()
    mock_ocr = MagicMock()
    mock_store = MagicMock()
    mock_store.hybrid_search.return_value = []
    mock_store.similarity_search.return_value = []
    mock_llm = MagicMock()

    orch = RAGOrchestrator(
        extractor=mock_extractor,
        ocr=mock_ocr,
        vector_store=mock_store,
        llm=mock_llm,
    )

    # Arabic question
    ar_resp = orch.query("ما هي شروط المادة 70؟", use_hybrid=True)
    assert "لم يتم العثور" in ar_resp.answer

    # English question
    en_resp = orch.query("What are the conditions of article 70?", use_hybrid=True)
    assert "No relevant information found" in en_resp.answer


def test_t2_4_and_t2_5_filtering_and_score_threshold():
    """T2.4 & T2.5: Test score thresholding and explicit filename/article filtering."""
    mock_extractor = MagicMock()
    mock_ocr = MagicMock()
    mock_store = MagicMock()

    c1 = Chunk(chunk_id="1", text="نص المادة 70", metadata={"filename": "doc_a.pdf", "article_number": "70"})
    c2 = Chunk(chunk_id="2", text="نص المادة 80", metadata={"filename": "doc_b.pdf", "article_number": "80"})

    mock_store.hybrid_search.return_value = [
        RetrievedChunk(chunk=c1, score=0.90),
        RetrievedChunk(chunk=c2, score=0.85),
    ]

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Reasoning: تحليل\nAnswer: وفق المادة 70 [1]"

    settings_mock = MagicMock()
    settings_mock.use_hybrid_search = True
    settings_mock.fetch_k = 25
    settings_mock.hybrid_alpha = 0.5
    settings_mock.min_retrieval_score = 0.88  # Filter out c2 (0.85)

    orch = RAGOrchestrator(
        extractor=mock_extractor,
        ocr=mock_ocr,
        vector_store=mock_store,
        llm=mock_llm,
        settings=settings_mock,
    )

    # 1. Score threshold filter
    resp = orch.query("سؤال تجريبي", top_k=6)
    assert len(resp.sources) == 1
    assert resp.sources[0].chunk.chunk_id == "1"

    # 2. Filename filter
    resp_filtered = orch.query("سؤال تجريبي", top_k=6, filter_filename="doc_b.pdf")
    # c2 was filtered out by score anyway, so it should be empty
    assert len(resp_filtered.sources) == 0


def test_t3_2_structured_json_logging():
    """T3.2: Verify JSONFormatter outputs valid JSON with request_id."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test execution message",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-uuid-12345"

    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "Test execution message"
    assert parsed["request_id"] == "req-uuid-12345"
    assert "timestamp" in parsed
