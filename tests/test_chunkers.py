"""
tests/test_chunkers.py for RAG_XPER
"""
import pytest
from rag_xper.core.models import PageContent, SourceType
from rag_xper.core.ingestion.text_chunker import (
    RecursiveChunker,
    ParentChildChunker,
    ArticleBasedChunker,
    AutoDetectChunker,
    ChunkerFactory,
)


def test_recursive_chunker():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    sample_text = "الفقرة الأولى تتحدث عن بناء العادات.\n\nالفقرة الثانية تشرح كيفية تقليل المشتتات."
    page = PageContent(source_path="test.pdf", page_number=1, text=sample_text, source_type=SourceType.NATIVE_TEXT)

    chunks = chunker.chunk_pages([page])
    assert len(chunks) >= 1
    assert chunks[0].metadata["strategy"] == "recursive"
    assert "content_hash" in chunks[0].metadata


def test_parent_child_chunker():
    chunker = ParentChildChunker(parent_chunk_size=300, child_chunk_size=80, child_overlap=20)
    long_text = (
        "هذا نص طويل لتجربة التقطيع الهرمي Parent-Child في نظام RAG_XPER. "
        "نقوم باختبار تقسيم النص إلى أجزاء كبيرة للنموذج وأجزاء صغيرة للبحث المتجهي. "
        "تساعد هذه الاستراتيجية في تحسين دقة الاسترجاع بشكل ملحوظ."
    )
    page = PageContent(source_path="legal.pdf", page_number=1, text=long_text, source_type=SourceType.NATIVE_TEXT)

    chunks = chunker.chunk_pages([page])
    assert len(chunks) >= 2
    for c in chunks:
        assert c.metadata["is_child"] is True
        assert "parent_id" in c.metadata
        assert "parent_text" in c.metadata
        assert len(c.metadata["parent_text"]) >= len(c.text)


def test_article_based_chunker():
    chunker = ArticleBasedChunker(fallback_max_size=500)
    legal_text = (
        "المادة الأولى:\nيجب على كل طرف تقديم أدلته كتابة.\n\n"
        "المادة الثانية:\nتعتبر المحررات الرسمية حجة قاطعة.\n\n"
        "المادة الثالثة:\nلا يجوز إثبات التصرفات التي تزيد عن مائة ألف ريال بشهادة الشهود."
    )
    page = PageContent(source_path="law.pdf", page_number=5, text=legal_text, source_type=SourceType.NATIVE_TEXT)

    chunks = chunker.chunk_pages([page])
    assert len(chunks) >= 3


def test_chunker_factory():
    c1 = ChunkerFactory.create_chunker("recursive")
    assert isinstance(c1, RecursiveChunker)

    c2 = ChunkerFactory.create_chunker("parent_child")
    assert isinstance(c2, ParentChildChunker)

    c3 = ChunkerFactory.create_chunker("article_based")
    assert isinstance(c3, ArticleBasedChunker)

    c4 = ChunkerFactory.create_chunker("auto")
    assert isinstance(c4, AutoDetectChunker)
