"""
rag_xper.bootstrap

Single, unified wiring point that builds and connects all components of the RAG pipeline.
Decouples core initialization from CLI, Web UI, and REST API.
"""
from __future__ import annotations

from rag_xper.config import settings
from rag_xper.core.generation.llm_interface import GeminiLLM, OllamaLLM
from rag_xper.core.generation.rag_orchestrator import RAGOrchestrator
from rag_xper.core.ingestion.document_extractor import DocumentExtractor
from rag_xper.core.ingestion.ocr_engine import OCREngine
from rag_xper.core.retrieval.vector_store_manager import VectorStoreFactory
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)


def build_orchestrator(custom_settings=None) -> RAGOrchestrator:
    """Instantiate and wire all pipeline components together."""
    active_settings = custom_settings or settings
    active_settings.validate()

    # 1. LLM Client
    if active_settings.llm_provider == "gemini":
        llm = GeminiLLM(
            api_key=active_settings.gemini_api_key,
            model_name=active_settings.gemini_model,
            embedding_model=active_settings.gemini_embedding_model,
            timeout_seconds=active_settings.llm_timeout_seconds,
            max_retries=active_settings.llm_max_retries,
        )
    else:
        llm = OllamaLLM(
            base_url=active_settings.ollama_base_url,
            model_name=active_settings.ollama_model,
            embedding_model=active_settings.ollama_embedding_model,
            timeout_seconds=active_settings.llm_timeout_seconds,
            max_retries=active_settings.llm_max_retries,
        )

    # 2. Vector Store (Qdrant by default or ChromaDB)
    vector_store = VectorStoreFactory.create_vector_store(
        config=active_settings,
        embedding_fn=llm.embed,
    )

    # 3. Document Extractor & OCR Engine
    extractor = DocumentExtractor(
        native_text_min_chars=active_settings.native_text_min_chars,
        render_zoom=active_settings.ocr_render_zoom,
    )
    ocr = OCREngine(
        engine=active_settings.ocr_engine,
        languages=list(active_settings.ocr_languages),
    )

    return RAGOrchestrator(
        extractor=extractor,
        ocr=ocr,
        vector_store=vector_store,
        llm=llm,
        settings=active_settings,
    )
