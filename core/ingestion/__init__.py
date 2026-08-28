"""
core/ingestion package for RAG_XPER.

Contains document parsers, OCR engine, and modular chunking strategies.
"""
from core.ingestion.document_extractor import DocumentExtractor
from core.ingestion.ocr_engine import OCREngine
from core.ingestion.text_chunker import (
    ArticleBasedChunker,
    AutoDetectChunker,
    BaseChunker,
    ChunkerFactory,
    ParentChildChunker,
    RecursiveChunker,
)

__all__ = [
    "DocumentExtractor",
    "OCREngine",
    "BaseChunker",
    "RecursiveChunker",
    "ParentChildChunker",
    "ArticleBasedChunker",
    "AutoDetectChunker",
    "ChunkerFactory",
]
