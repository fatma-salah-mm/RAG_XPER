"""
tests/test_chunkers.py for RAG_XPER
"""
import pytest
from core.ingestion import (
    ArticleBasedChunker,
    AutoDetectChunker,
    ChunkerFactory,
    ParentChildChunker,
    RecursiveChunker,
)
from core.models import PageContent, SourceType


def test_recursive_chunker():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    sample_text = "الفقرة الأولى تتحدث عن أهمية بناء العادات الإيجابية في الحياة اليومية.\n\nالفقرة الثانية تشرح كيفية تقليل المشتتات والتركيز على الأهداف الأساسية."
    page = PageContent(source_path="test.pdf", page_number=1, text=sample_text, source_type=SourceType.NATIVE_TEXT)

    chunks = chunker.chunk_pages([page])
    assert len(chunks) >= 1
    assert chunks[0].metadata["strategy"] == "recursive"


def test_parent_child_chunker():
    chunker = ParentChildChunker(parent_chunk_size=300, child_chunk_size=80, child_overlap=20)
    long_text = (
        "هذا نص طويل لتجربة التقطيع الهرمي Parent-Child في نظام RAG_XPER. "
        "نقوم باختبار تقسيم النص إلى أجزاء كبيرة للنموذج وأجزاء صغيرة للبحث المتجهي. "
        "تساعد هذه الاستراتيجية في تحسين دقة الاسترجاع بشكل ملحوظ دون فقدان السياق الكامل للمادة أو الفصل."
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
        "نظام الإثبات التجريبي:\n\n"
        "المادة الأولى:\nيجب على كل طرف تقديم أدلته كتابة.\n\n"
        "المادة الثانية:\nتعتبر المحررات الرسمية حجة قاطعة على الكافة بما دون فيها.\n\n"
        "المادة الثالثة:\nلا يجوز إثبات التصرفات التي تزيد عن مائة ألف ريال بشهادة الشهود إلا باستثناء."
    )
    page = PageContent(source_path="law.pdf", page_number=5, text=legal_text, source_type=SourceType.NATIVE_TEXT)

    chunks = chunker.chunk_pages([page])
    assert len(chunks) >= 3
    articles_found = [c.text for c in chunks if "المادة" in c.text]
    assert len(articles_found) >= 3


def test_chunker_factory():
    c1 = ChunkerFactory.create_chunker("recursive")
    assert isinstance(c1, RecursiveChunker)

    c2 = ChunkerFactory.create_chunker("parent_child")
    assert isinstance(c2, ParentChildChunker)

    c3 = ChunkerFactory.create_chunker("article_based")
    assert isinstance(c3, ArticleBasedChunker)

    c4 = ChunkerFactory.create_chunker("auto")
    assert isinstance(c4, AutoDetectChunker)
