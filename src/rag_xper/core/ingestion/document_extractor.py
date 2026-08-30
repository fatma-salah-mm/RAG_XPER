"""
rag_xper.core.ingestion.document_extractor

Extracts text and renders raster images from PDF, Markdown, and Text documents.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from rag_xper.core.exceptions import DocumentExtractionError
from rag_xper.core.models import PageContent, SourceType
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)

_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}


class DocumentExtractor:
    """Extracts raw text or renders pages from documents."""

    def __init__(self, native_text_min_chars: int = 20, render_zoom: float = 2.5) -> None:
        self._min_chars = native_text_min_chars
        self._zoom = render_zoom

    def extract(self, file_path: str) -> List[PageContent]:
        """Parse a PDF, Markdown, or text file and return a list of PageContent."""
        path = Path(file_path)
        if not path.exists():
            raise DocumentExtractionError(f"File not found: {file_path}")

        # Direct text / Markdown ingestion (Phase 1.5)
        if path.suffix.lower() in _TEXT_EXTENSIONS:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                logger.info("Directly extracted text from '%s' (%d chars)", path.name, len(content))
                return [
                    PageContent(
                        source_path=str(path),
                        page_number=1,
                        text=content,
                        source_type=SourceType.MARKDOWN if path.suffix.lower() in {".md", ".markdown"} else SourceType.NATIVE_TEXT,
                    )
                ]
            except Exception as exc:
                raise DocumentExtractionError(f"Failed to read text file '{file_path}': {exc}") from exc

        # PDF Ingestion
        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise DocumentExtractionError(f"Failed to open PDF '{file_path}': {exc}") from exc

        pages: List[PageContent] = []
        try:
            for page_num in range(1, len(doc) + 1):
                page = doc[page_num - 1]
                native_text = page.get_text("text").strip()

                if len(native_text) >= self._min_chars:
                    pages.append(
                        PageContent(
                            source_path=str(path),
                            page_number=page_num,
                            text=native_text,
                            source_type=SourceType.NATIVE_TEXT,
                        )
                    )
                else:
                    mat = fitz.Matrix(self._zoom, self._zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    image_bytes = pix.tobytes("png")
                    pages.append(
                        PageContent(
                            source_path=str(path),
                            page_number=page_num,
                            text="",
                            source_type=SourceType.OCR,
                            raw_image=image_bytes,
                        )
                    )
        finally:
            doc.close()

        logger.info(
            "Extracted %d pages from '%s' (%d native, %d OCR)",
            len(pages),
            path.name,
            sum(1 for p in pages if p.source_type == SourceType.NATIVE_TEXT),
            sum(1 for p in pages if p.source_type == SourceType.OCR),
        )
        return pages
