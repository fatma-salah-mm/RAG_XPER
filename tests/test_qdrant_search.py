"""
tests/test_qdrant_search.py for RAG_XPER
"""
import shutil
import tempfile
import numpy as np
import pytest
from core.models import Chunk
from core.retrieval.qdrant_store_manager import QdrantStoreManager, _QDRANT_AVAILABLE


@pytest.mark.skipif(not _QDRANT_AVAILABLE, reason="qdrant-client not installed")
def test_qdrant_store_manager():
    temp_dir = tempfile.mkdtemp()

    def dummy_embed(texts):
        # deterministic 8-dimensional dummy embeddings
        vecs = []
        for t in texts:
            val = float(len(t) % 10) / 10.0
            vecs.append([val] * 8)
        return vecs

    try:
        store = QdrantStoreManager(
            storage_path=temp_dir,
            collection_name="test_collection",
            embedding_dim=8,
            embedding_fn=dummy_embed,
        )

        c1 = Chunk(
            chunk_id="doc1:p1:c0",
            text="المادة السبعون تنص على عدم جواز شهادة الشهود لما زاد عن مائة ألف ريال",
            metadata={"source": "law.pdf", "page": 70, "strategy": "article_based"},
        )
        c2 = Chunk(
            chunk_id="doc1:p1:c1",
            text="المحرر الرسمي حجة على الكافة بما دون فيه من أفعال موظف عام",
            metadata={"source": "law.pdf", "page": 25, "strategy": "article_based"},
        )

        # Upsert
        count = store.upsert_chunks([c1, c2])
        assert count == 2

        # Hybrid Search
        results = store.hybrid_search("المادة السبعون", top_k=2)
        assert len(results) >= 1
        assert "السبعون" in results[0].chunk.text

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
