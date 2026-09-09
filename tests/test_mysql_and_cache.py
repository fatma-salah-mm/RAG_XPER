"""
tests/test_mysql_and_cache.py

Unit and Integration Tests for In-Memory QueryCache and MySQL Database Layer.
"""
from __future__ import annotations

import time
import pytest
from fastapi.testclient import TestClient

from rag_xper.api.app import app
from rag_xper.core.cache import QueryCache
from rag_xper.core.db.service import delete_book, get_query_history, list_books, log_query, register_book
from rag_xper.core.db.session import init_db
from rag_xper.core.models import Chunk, RAGResponse, RetrievedChunk

client = TestClient(app)


def test_query_cache_hit_and_arabic_normalization():
    """Verify cache matches Arabic spelling variations (Alef, Taa Marbuta)."""
    cache = QueryCache(max_size=10, ttl_seconds=60)
    dummy_resp = RAGResponse(
        query="ما هي شروط المادة 70؟",
        answer="شروط قبول الشهادة وفق المادة 70...",
        sources=[RetrievedChunk(chunk=Chunk(chunk_id="1", text="نص المادة 70"), score=0.95)],
        reasoning="تحليل قانوني",
    )

    # Cache with standard form
    cache.set("ما هي شروط المادة 70؟", dummy_resp, top_k=6)

    # Query with Taa Marbuta / diacritics variations
    hit1 = cache.get("ما هي شروط الماده 70؟", top_k=6)
    assert hit1 is not None
    assert hit1.answer == dummy_resp.answer

    hit2 = cache.get("مَا هِيَ شُرُوطُ الْمَادَّةِ 70؟", top_k=6)
    assert hit2 is not None

    # Miss on different question
    miss = cache.get("سؤال مختلف تماماً", top_k=6)
    assert miss is None


def test_query_cache_ttl_expiration():
    """Verify cache entries expire after TTL."""
    cache = QueryCache(max_size=10, ttl_seconds=1)  # 1 second TTL
    dummy_resp = RAGResponse(query="سؤال مؤقت", answer="إجابة مؤقتة", sources=[], reasoning="تحليل تجريبي")

    cache.set("سؤال مؤقت", dummy_resp, top_k=6)
    assert cache.get("سؤال مؤقت", top_k=6) is not None

    time.sleep(1.1)
    assert cache.get("سؤال مؤقت", top_k=6) is None


def test_mysql_books_catalog_crud():
    """Verify books registration, retrieval, and deletion in database."""
    init_db()

    # Register
    book = register_book(
        title="نظام المعاملات المدنية السعودي",
        filename="civil_transactions.pdf",
        file_path="./data/documents/civil_transactions.pdf",
        author="وزارة العدل",
        category="أنظمة وقوانين",
        chunk_count=120,
        strategy_used="article_based",
    )
    assert book is not None
    assert book.id is not None

    # List
    books = list_books(category="أنظمة وقوانين")
    assert any(b["filename"] == "civil_transactions.pdf" for b in books)

    # Delete
    deleted = delete_book("civil_transactions.pdf")
    assert deleted is True

    books_after = list_books(category="أنظمة وقوانين")
    assert not any(b["filename"] == "civil_transactions.pdf" for b in books_after)


def test_mysql_query_logging():
    """Verify logging questions and telemetry to query_logs table."""
    init_db()
    log_query(
        question="ما هي أحكام العقد؟",
        answer="العقد شريعة المتعاقدين",
        reasoning="قاعدة فقهية ونظامية",
        sources=[{"source": "law.pdf", "page": 5}],
        execution_time_ms=12.5,
        is_cached=False,
        session_id="test_session_123",
    )

    history = get_query_history(session_id="test_session_123")
    assert len(history) >= 1
    assert history[0]["question"] == "ما هي أحكام العقد؟"


def test_api_books_and_ui_endpoints():
    """Verify /v1/books and /ui static dashboard endpoints."""
    # /v1/books
    res_books = client.get("/v1/books")
    assert res_books.status_code == 200
    assert isinstance(res_books.json(), list)

    # /ui/
    res_ui = client.get("/ui/")
    assert res_ui.status_code == 200
    assert "XPER" in res_ui.text
    assert "المساعد الذكي" in res_ui.text
