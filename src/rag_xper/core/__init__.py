"""
rag_xper.core package

Enterprise Multi-Modal RAG Engine:
- Ingestion: Document Extraction (PDF, MD, TXT), OCR, Modular Chunkers
- Retrieval: Qdrant Server/Embedded, ChromaDB, Persistent BM25, Shared RRF
- Generation: Chain-of-Thought LLM Gateway (Gemini, Ollama), Orchestrator
"""
from rag_xper.core.exceptions import (
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
from rag_xper.core.generation import BaseLLM, GeminiLLM, OllamaLLM, RAGOrchestrator
from rag_xper.core.ingestion import (
    ArticleBasedChunker,
    AutoDetectChunker,
    BaseChunker,
    ChunkerFactory,
    DocumentExtractor,
    OCREngine,
    ParentChildChunker,
    RecursiveChunker,
)
from rag_xper.core.models import Chunk, PageContent, RAGResponse, RetrievedChunk, SourceType
from rag_xper.core.retrieval import (
    BaseVectorStore,
    BM25Retriever,
    ChromaVectorStoreManager,
    QdrantStoreManager,
    VectorStoreFactory,
    reciprocal_rank_fusion,
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
    "reciprocal_rank_fusion",
    "QdrantStoreManager",
    "ChromaVectorStoreManager",
    "VectorStoreFactory",
    # Generation
    "BaseLLM",
    "GeminiLLM",
    "OllamaLLM",
    "RAGOrchestrator",
]
