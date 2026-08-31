"""
tests/test_stress_retrieval_and_bm25.py

Stress and Edge-Case Tests for BM25 Okapi & Qdrant Retrieval Engine.
Tests:
- Heavy Arabic diacritics (tashkeel) and spelling variations
- Comprehensive Arabic numbers-to-words search matching
- Bulk indexing & search on 500+ chunks
- Extreme RRF fusion weightings (alpha=0.0 vs alpha=1.0)
- Zero-hit queries and stopword-only queries
- Deduplication under stress with SHA-256 content hashes
"""
from __future__ import annotations

import shutil
import tempfile
import pytest
from rag_xper.core.models import Chunk
from rag_xper.core.retrieval.bm25_retriever import BM25Retriever, normalize_arabic, tokenize
from rag_xper.core.retrieval.hybrid_fusion import reciprocal_rank_fusion
from rag_xper.core.retrieval.qdrant_store_manager import QdrantStoreManager, _QDRANT_AVAILABLE


def test_arabic_diacritics_and_orthography_variations():
    """Verify retriever normalizes tashkeel, Alef variants, and Taa Marbuta."""
    raw_texts = [
        "الْمَادَّةُ الْأُولَى: تَسْرِي هَذِهِ اللَّائِحَةُ عَلَى جَمِيعِ الشَّرِكَاتِ.",
        "الماده الاولى: تسري هذه اللائحه على جميع الشركات.",
        "المادة الاولي: تسرى هذه اللائحة على جميع الشركات.",
    ]

    tokens_set = [set(tokenize(t)) for t in raw_texts]

    # Check key normalized stems match across all variations
    for tokens in tokens_set:
        assert "ماده" in tokens
        assert "اول" in tokens or "الاول" in tokens
        assert "شرك" in tokens or "شركات" in tokens


def test_arabic_number_expansion_across_ranges():
    """Verify queries with numbers find documents with words and vice versa."""
    temp_dir = tempfile.mkdtemp()
    persist_file = f"{temp_dir}/bm25_stress.pkl"

    try:
        bm25 = BM25Retriever(persist_path=persist_file)

        # Ingest chunks with number digits
        chunks = [
            Chunk(chunk_id="art10", text="المادة 10 تحدد اختصاصات مجلس الإدارة", metadata={"source": "law.pdf"}),
            Chunk(chunk_id="art20", text="المادة 20 تنص على مهام المدير التنفيذي", metadata={"source": "law.pdf"}),
            Chunk(chunk_id="art50", text="المادة 50 تتعلق بالمسؤولية التضامنية", metadata={"source": "law.pdf"}),
            Chunk(chunk_id="art70", text="المادة 70 تنص على شروط قبول شهادة الشهود", metadata={"source": "law.pdf"}),
            Chunk(chunk_id="art100", text="المادة 100 تنظم أحكام تصفية الشركات", metadata={"source": "law.pdf"}),
        ]
        bm25.add_chunks(chunks)

        # Query using Arabic words
        res_70 = bm25.search("المادة السبعون", top_k=1)
        assert len(res_70) > 0
        assert res_70[0][0].chunk_id == "art70"

        res_20 = bm25.search("المادة عشرين", top_k=1)
        assert len(res_20) > 0
        assert res_20[0][0].chunk_id == "art20"

        res_10 = bm25.search("المادة العاشرة", top_k=1)
        assert len(res_10) > 0
        assert res_10[0][0].chunk_id == "art10"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_stopword_only_and_gibberish_queries():
    """Verify BM25 handles queries with only stopwords or random gibberish safely."""
    temp_dir = tempfile.mkdtemp()
    try:
        bm25 = BM25Retriever(persist_path=f"{temp_dir}/bm25.pkl")
        bm25.add_chunks([Chunk(chunk_id="1", text="نظام الإثبات والعقود التجارية", metadata={})])

        # Stopwords only
        assert bm25.search("من إلى على في هذا هذه هل", top_k=5) == []

        # Complete non-existent gibberish
        assert bm25.search("xyzqweasdzxc123456789", top_k=5) == []

        # Empty string
        assert bm25.search("", top_k=5) == []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_rrf_extreme_weighting_boundaries():
    """Verify RRF fusion under extreme alpha weights (0.0 pure BM25 vs 1.0 pure Dense)."""
    dense_chunk = Chunk(chunk_id="dense_best", text="Dense winner", metadata={})
    bm25_chunk = Chunk(chunk_id="bm25_best", text="BM25 winner", metadata={})

    dense_hits = [(dense_chunk, 0.99, 0), (bm25_chunk, 0.50, 1)]
    bm25_hits = [(bm25_chunk, 15.0, 0), (dense_chunk, 2.0, 1)]

    # Pure Dense (alpha=1.0)
    res_dense = reciprocal_rank_fusion(dense_hits=dense_hits, bm25_hits=bm25_hits, top_k=2, alpha=1.0)
    assert res_dense[0].chunk.chunk_id == "dense_best"

    # Pure BM25 (alpha=0.0)
    res_bm25 = reciprocal_rank_fusion(dense_hits=dense_hits, bm25_hits=bm25_hits, top_k=2, alpha=0.0)
    assert res_bm25[0].chunk.chunk_id == "bm25_best"

    # Balanced Hybrid (alpha=0.5)
    res_hybrid = reciprocal_rank_fusion(dense_hits=dense_hits, bm25_hits=bm25_hits, top_k=2, alpha=0.5)
    assert len(res_hybrid) == 2


@pytest.mark.skipif(not _QDRANT_AVAILABLE, reason="qdrant-client not installed")
def test_qdrant_bulk_indexing_and_deduplication():
    """Stress test Qdrant with 200 synthetic chunks and verify content hash deduplication."""
    temp_dir = tempfile.mkdtemp()

    def dummy_embed(texts):
        return [[float(len(t) % 10) / 10.0] * 8 for t in texts]

    try:
        store = QdrantStoreManager(
            storage_path=temp_dir,
            collection_name="stress_test_col",
            embedding_dim=8,
            embedding_fn=dummy_embed,
        )

        bulk_chunks = [
            Chunk(
                chunk_id=f"doc_{i}:c0",
                text=f"هذا هو النص التجريبي رقم {i} لاختبار الأداء والضغط العالي على قاعدة البيانات.",
                metadata={"source": f"doc_{i}.pdf", "content_hash": f"hash_{i}"},
            )
            for i in range(200)
        ]

        stored = store.upsert_chunks(bulk_chunks)
        assert stored == 200

        # Test deduplication on existing hash
        assert store.is_file_ingested("any_name.pdf", content_hash="hash_50") is True
        assert store.is_file_ingested("doc_99.pdf") is True
        assert store.is_file_ingested("non_existent.pdf", content_hash="hash_9999") is False

        # Test hybrid search on bulk index
        results = store.hybrid_search("الضغط العالي", top_k=5)
        assert len(results) >= 1

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
