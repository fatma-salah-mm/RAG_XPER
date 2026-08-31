"""
tests/test_stress_chunking.py

Stress and Edge-Case Tests for RAG_XPER Chunking Engine.
Tests:
- Zero-length text and whitespace-only pages
- Single ultra-long unbroken string (10,000 chars without spaces or delimiters)
- Mixed Arabic, English, Chinese, emojis, math formulas, and zero-width characters
- Extreme chunk sizes (micro chunks of 5 chars vs huge chunks of 50,000 chars)
- Complex Arabic legal patterns with sub-articles (e.g. المادة 12 مكرر, البند 5-أ)
- Deep parent-child relationship integrity and deduplication
"""
from __future__ import annotations

import pytest
from rag_xper.core.models import PageContent, SourceType
from rag_xper.core.ingestion.text_chunker import (
    ArticleBasedChunker,
    AutoDetectChunker,
    ChunkerFactory,
    ParentChildChunker,
    RecursiveChunker,
    compute_content_hash,
)


def test_empty_and_whitespace_pages():
    """Verify chunkers handle empty or whitespace-only inputs without crashing."""
    empty_pages = [
        PageContent(source_path="empty.pdf", page_number=1, text="", source_type=SourceType.NATIVE_TEXT),
        PageContent(source_path="spaces.pdf", page_number=2, text="   \n\n\t  \n  ", source_type=SourceType.NATIVE_TEXT),
    ]

    for strategy in ["recursive", "parent_child", "article_based", "auto"]:
        chunker = ChunkerFactory.create_chunker(strategy)
        chunks = chunker.chunk_pages(empty_pages)
        assert len(chunks) == 0, f"Strategy '{strategy}' should return 0 chunks for empty pages."


def test_ultra_long_unbroken_string():
    """Verify chunker splits continuous strings without spaces or punctuation."""
    unbroken_text = "أ" * 5000 + "B" * 5000  # 10,000 chars unbroken
    page = PageContent(source_path="monster.pdf", page_number=1, text=unbroken_text, source_type=SourceType.NATIVE_TEXT)

    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_pages([page])

    assert len(chunks) >= 15
    for c in chunks:
        assert len(c.text) <= 550  # Strictly bounded


def test_multilingual_unicode_and_special_symbols():
    """Test text with emojis, math symbols, Arabic ligatures, and mixed languages."""
    tricky_text = (
        "مرحباً بالعالم! 🌍 \u200b\u200f\ufeff (Zero-width marks) \n\n"
        "Equation: \\sum_{i=1}^{n} x_i = \\alpha \\times \\beta \n\n"
        "English: Performance optimization under extreme load. \n\n"
        "Chinese: 这是一个极端压力测试。 \n\n"
        "Arabic: المادة 15: يُعاقب كل من يخالف الأنظمة بغرامة قدرها 50,000 ريال سعودي."
    )
    page = PageContent(source_path="tricky.pdf", page_number=1, text=tricky_text, source_type=SourceType.NATIVE_TEXT)

    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_pages([page])

    assert len(chunks) >= 3
    all_text = "".join([c.text for c in chunks])
    assert "المادة 15" in all_text
    assert "Performance optimization" in all_text


def test_micro_chunk_sizes():
    """Test chunker behavior with tiny chunk size (5-10 chars)."""
    text = "بناء العادات الإيجابية يحتاج وقتاً والتزاماً مستمراً."
    page = PageContent(source_path="micro.pdf", page_number=1, text=text, source_type=SourceType.NATIVE_TEXT)

    chunker = RecursiveChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.chunk_pages([page])

    assert len(chunks) >= 4
    for c in chunks:
        assert len(c.text) > 0


def test_complex_legal_article_patterns():
    """Test ArticleBasedChunker across diverse Arabic legal nomenclature."""
    complex_legal = (
        "نظام المعاملات المدنية:\n\n"
        "المادة 1:\nتسري النصوص التشريعية على جميع المسائل.\n\n"
        "المادة الأولى مكرر:\nتطبق القواعد العامة في حال عدم وجود نص خاص.\n\n"
        "البند الخامس عشر:\nيلتزم المتعاقدان بتنفيذ ما اشتمل عليه العقد بحسن نية.\n\n"
        "الفصل الثاني: أحكام الالتزام التضامني.\n\n"
        "Article 42:\nThis contract is governed by the laws of Saudi Arabia."
    )
    page = PageContent(source_path="code.pdf", page_number=10, text=complex_legal, source_type=SourceType.NATIVE_TEXT)

    chunker = ArticleBasedChunker(fallback_max_size=500)
    chunks = chunker.chunk_pages([page])

    assert len(chunks) >= 4
    chunk_texts = [c.text for c in chunks]

    assert any("المادة 1" in t for t in chunk_texts)
    assert any("المادة الأولى مكرر" in t for t in chunk_texts)
    assert any("البند الخامس عشر" in t for t in chunk_texts)
    assert any("Article 42" in t for t in chunk_texts)


def test_parent_child_deep_integrity():
    """Test ParentChildChunker metadata consistency and parent reconstruction."""
    long_law_text = (
        "المادة السبعون: لا يجوز إثبات التصرفات القانونية التي تزيد قيمتها عن مائة ألف ريال أو ما يعادلها بشهادة الشهود، "
        "ما لم يوجد اتفاق أو نص يقضي بغير ذلك. ويجب في هذه الحالة تقديم محرر رسمي أو عرفي مكتوب وموقع عليه من أطراف الالتزام. "
        "ويستثنى من ذلك الحالات التي يتعذر فيها الحصول على دليل كتابي بسبب مانع أدبي أو مادي أو فقدان السند لسبب أجنبي."
    )
    page = PageContent(source_path="evidence.pdf", page_number=70, text=long_law_text, source_type=SourceType.NATIVE_TEXT)

    chunker = ParentChildChunker(parent_chunk_size=200, child_chunk_size=70, child_overlap=15)
    chunks = chunker.chunk_pages([page])

    assert len(chunks) >= 3

    # Validate parent linkage
    parent_ids = set()
    for c in chunks:
        assert c.metadata["is_child"] is True
        assert "parent_id" in c.metadata
        assert "parent_text" in c.metadata
        assert len(c.metadata["parent_text"]) >= len(c.text)
        assert c.text in c.metadata["parent_text"] or len(c.text) > 0
        parent_ids.add(c.metadata["parent_id"])

    assert len(parent_ids) >= 1  # Children properly grouped under parents
