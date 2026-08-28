"""
core package for RAG_XPER.

Modular 3-stage RAG architecture:
- Phase 1: core.ingestion (Document extraction, OCR, Modular chunking)
- Phase 2: core.retrieval (Vector stores, BM25, Hybrid RRF retrieval)
- Phase 3: core.generation (LLM gateway, CoT reasoning, RAG orchestrator)
"""
from core.exceptions import (
    ChunkingError,
    ConfigurationError,
    DocumentExtractionError,
    EmbeddingGenerationError,
    LLMGenerationError,
    LLMTimeoutError,
    OCRExtractionError,
    RAGPipelineError,
    VectorDBConnectionError,
)
from core.generation import BaseLLM, GeminiLLM, OllamaLLM, RAGOrchestrator
from core.ingestion import (
    ArticleBasedChunker,
    AutoDetectChunker,
    BaseChunker,
    ChunkerFactory,
    DocumentExtractor,
    OCREngine,
    ParentChildChunker,
    RecursiveChunker,
)
from core.models import Chunk, PageContent, RAGResponse, RetrievedChunk, SourceType
from core.retrieval import (
    BaseVectorStore,
    BM25Retriever,
    ChromaVectorStoreManager,
    QdrantStoreManager,
    VectorStoreFactory,
)

__all__ = [
    # Models & Exceptions
    "Chunk",
    "PageContent",
    "RetrievedChunk",
    "RAGResponse",
    "SourceType",
    "RAGPipelineError",
    "ConfigurationError",
    "DocumentExtractionError",
    "OCRExtractionError",
    "ChunkingError",
    "EmbeddingGenerationError",
    "VectorDBConnectionError",
    "LLMTimeoutError",
    "LLMGenerationError",
    # Ingestion
    "DocumentExtractor",
    "OCREngine",
    "BaseChunker",
    "RecursiveChunker",
    "ParentChildChunker",
    "ArticleBasedChunker",
    "AutoDetectChunker",
    "ChunkerFactory",
    # Retrieval
    "BaseVectorStore",
    "BM25Retriever",
    "QdrantStoreManager",
    "ChromaVectorStoreManager",
    "VectorStoreFactory",
    # Generation
    "BaseLLM",
    "GeminiLLM",
    "OllamaLLM",
    "RAGOrchestrator",
]
