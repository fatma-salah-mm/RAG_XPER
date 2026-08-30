"""
tests/test_bm25_arabic.py for RAG_XPER
"""
import shutil
import tempfile
from rag_xper.core.models import Chunk
from rag_xper.core.retrieval.bm25_retriever import BM25Retriever, normalize_arabic, tokenize


def test_arabic_normalization():
    assert normalize_arabic("أحمد") == "احمد"
    assert normalize_arabic("مكتبة") == "مكتبه"
    assert normalize_arabic("الْقَاضِي") == "القاضي"


def test_bm25_search_and_persistence():
    temp_dir = tempfile.mkdtemp()
    persist_file = f"{temp_dir}/bm25_test.pkl"
    try:
        bm25 = BM25Retriever(persist_path=persist_file)
        c1 = Chunk(chunk_id="1", text="المادة 70 تنص على شروط قبول الشهادة في المحاكم", metadata={"source": "law.pdf"})
        c2 = Chunk(chunk_id="2", text="العادات الذرية كتاب في تطوير الذات وبناء السلوكيات", metadata={"source": "habits.pdf"})

        bm25.add_chunks([c1, c2])

        # Test search with Arabic number-to-words expansion (70 -> سبعون / سبعين)
        results = bm25.search("المادة السبعون", top_k=2)
        assert len(results) >= 1
        assert results[0][0].chunk_id == "1"

        # Test persistence reload
        bm25_reloaded = BM25Retriever(persist_path=persist_file)
        assert len(bm25_reloaded._chunks) == 2
        results_reloaded = bm25_reloaded.search("العادات الذرية", top_k=1)
        assert results_reloaded[0][0].chunk_id == "2"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
