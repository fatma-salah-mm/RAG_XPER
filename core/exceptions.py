"""
core/exceptions.py for RAG_XPER

Custom exception hierarchy for the RAG_XPER pipeline.
"""
from __future__ import annotations


class RAGPipelineError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(RAGPipelineError):
    """Raised when required configuration/environment variables are missing or invalid."""


class DocumentExtractionError(RAGPipelineError):
    """Raised when native document parsing fails."""


class OCRExtractionError(RAGPipelineError):
    """Raised when the OCR engine fails to initialise or process an image."""


class ChunkingError(RAGPipelineError):
    """Raised when the chunking strategy fails on a given input or is misconfigured."""


class EmbeddingGenerationError(RAGPipelineError):
    """Raised when the embedding provider fails to produce vectors."""


class VectorDBConnectionError(RAGPipelineError):
    """Raised when the vector database cannot be reached, initialised, or queried."""


class LLMTimeoutError(RAGPipelineError):
    """Raised when an LLM call exceeds the configured timeout."""


class LLMGenerationError(RAGPipelineError):
    """Raised for any other failure during LLM text generation."""
