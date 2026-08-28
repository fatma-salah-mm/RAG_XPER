"""
tests/test_orchestrator.py for RAG_XPER
"""
from core.generation import BaseLLM, RAGOrchestrator
from core.ingestion import DocumentExtractor, OCREngine
from core.models import Chunk, PageContent, RAGResponse, RetrievedChunk, SourceType
from core.retrieval import BaseVectorStore


class MockVectorStore(BaseVectorStore):
    def __init__(self):
        self.chunks = []

    def upsert_chunks(self, chunks):
        self.chunks.extend(chunks)
        return len(chunks)

    def similarity_search(self, query, top_k=4):
        return [RetrievedChunk(chunk=c, score=0.95) for c in self.chunks[:top_k]]

    def hybrid_search(self, query, top_k=6, fetch_k=25, alpha=0.5, rrf_k=60):
        return [RetrievedChunk(chunk=c, score=0.035) for c in self.chunks[:top_k]]

    def is_file_ingested(self, file_path):
        return False

    def delete_file(self, file_path):
        return 0


class MockLLM(BaseLLM):
    def generate(self, prompt: str) -> str:
        return "Reasoning: تم فحص المستند.\nAnswer: القيمة الإجمالية للعقد هي 500,000 ريال."

    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]


def test_rag_orchestrator_query():
    extractor = DocumentExtractor()
    ocr = OCREngine()
    vector_store = MockVectorStore()
    llm = MockLLM()

    # Preload chunk
    chunk = Chunk(
        chunk_id="test_doc:p1:c0",
        text="تنص المادة على أن القيمة الإجمالية للعقد هي 500,000 ريال سعودي.",
        metadata={"source": "contract.pdf", "page": 1, "source_type": "native_text"},
    )
    vector_store.upsert_chunks([chunk])

    orchestrator = RAGOrchestrator(
        extractor=extractor,
        ocr=ocr,
        vector_store=vector_store,
        llm=llm,
    )

    response = orchestrator.query("ما هي قيمة العقد؟")
    assert isinstance(response, RAGResponse)
    assert "500,000" in response.answer
    assert response.reasoning is not None
    assert len(response.sources) >= 1
