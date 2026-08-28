"""
core/models.py for RAG_XPER

Shared, strongly-typed data structures passed between pipeline stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class SourceType(str, Enum):
    """How a given page's text was obtained."""

    NATIVE_TEXT = "native_text"   # extracted directly from the PDF's text layer
    OCR = "ocr"                   # a scanned/image PDF page, resolved via OCR
    IMAGE_FILE = "image_file"     # a standalone image file, resolved via OCR


@dataclass
class PageContent:
    """Extracted content for a single page (or a standalone image)."""

    source_path: str
    page_number: int
    text: str
    source_type: SourceType
    confidence: Optional[float] = None
    raw_image: Optional[bytes] = field(default=None, repr=False)


@dataclass
class Chunk:
    """A chunk of text ready for embedding and vector-store persistence."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A chunk returned from vector search, carrying its similarity score."""

    chunk: Chunk
    score: float
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class RAGResponse:
    """Final, structured response returned by ``RAGOrchestrator.query``."""

    answer: str
    reasoning: Optional[str]
    sources: List[RetrievedChunk]
    query: str
    strategy_used: Optional[str] = None
