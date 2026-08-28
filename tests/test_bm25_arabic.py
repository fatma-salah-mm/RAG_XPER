"""
tests/test_bm25_arabic.py for RAG_XPER
"""
import pytest
from core.models import Chunk
from core.retrieval import (
    BM25Retriever,
    expand_query_tokens,
    normalize_arabic,
    tokenize,
)


def test_arabic_normalization():
    raw_text = "المَادَّةُ الأولى: الإثْبَاتُ بِالشَّهَادَةِ"
    normalized = normalize_arabic(raw_text)
    assert "الماده" in normalized
    assert "الاولي" in normalized
    assert "الشهاده" in normalized


def test_digit_to_ordinal_expansion():
    query = "المادة 70"
    tokens = expand_query_tokens(query)
    assert "70" in tokens
    assert any("سبع" in t for t in tokens)


def test_bm25_search_arabic():
    retriever = BM25Retriever()
    c1 = Chunk(
        chunk_id="doc1:p1",
        text="المادة السبعون تنص على عدم جواز الإثبات بشهادة الشهود في الالتزامات التعاقدية",
        metadata={"source": "law.pdf", "page": 70},
    )
    c2 = Chunk(
        chunk_id="doc1:p2",
        text="المادة الأولى تهدف لتنظيم إجراءات الإثبات في المعاملات المدنية والتجارية",
        metadata={"source": "law.pdf", "page": 1},
    )
    retriever.add_chunks([c1, c2])

    results = retriever.search("المادة 70", top_k=2)
    assert len(results) >= 1
    assert results[0][0].chunk_id == "doc1:p1"
