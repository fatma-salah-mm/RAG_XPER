"""
rag_xper.core.exceptions

Typed exception hierarchy for RAG_XPER.
"""
from __future__ import annotations


class RAGPipelineError(Exception):
    """Base class for all application-specific exceptions in RAG_XPER."""


class ConfigurationError(RAGPipelineError):
    """Raised when environment variables or settings are missing or invalid."""


class DocumentExtractionError(RAGPipelineError):
    """Raised when a PDF or document cannot be opened or parsed."""


class OCRExtractionError(RAGPipelineError):
    """Raised when the OCR engine fails to process an image."""


class ChunkingError(RAGPipelineError):
    """Raised when chunking fails to produce valid chunks."""


class EmbeddingGenerationError(RAGPipelineError):
    """Raised when the LLM provider fails to generate vector embeddings."""


class VectorDBConnectionError(RAGPipelineError):
    """Raised when Qdrant / ChromaDB cannot be reached or fails an operation."""


class LLMTimeoutError(RAGPipelineError):
    """Raised when an LLM generation call exceeds its configured timeout."""


class LLMGenerationError(RAGPipelineError):
    """Raised when the LLM returns an invalid or empty response."""
