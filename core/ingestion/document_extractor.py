"""
core/ingestion/document_extractor.py for RAG_XPER

Pure text extraction from digitally-native documents using PyMuPDF.
Rasterises scanned pages with high-DPI zoom (2.5) for OCR fallback.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pymupdf

from core.exceptions import DocumentExtractionError
from core.models import PageContent, SourceType
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentExtractor:
    """Extracts native (non-OCR) text from PDF documents, page by page."""

    def __init__(self, native_text_min_chars: int = 20, render_zoom: float = 2.5) -> None:
        self._min_chars = native_text_min_chars
        self._render_zoom = render_zoom

    def extract(self, file_path: str) -> List[PageContent]:
        path = Path(file_path)
        if not path.exists():
            raise DocumentExtractionError(f"File not found: {file_path}")

        try:
            doc = pymupdf.open(file_path)
        except Exception as exc:
            raise DocumentExtractionError(f"Failed to open PDF '{file_path}': {exc}") from exc

        pages: List[PageContent] = []
        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                text, ocr_image = self._extract_page(page, page_index)

                if ocr_image is not None:
                    pages.append(
                        PageContent(
                            source_path=str(path),
                            page_number=page_index + 1,
                            text="",
                            source_type=SourceType.OCR,
                            raw_image=ocr_image,
                        )
                    )
                else:
                    pages.append(
                        PageContent(
                            source_path=str(path),
                            page_number=page_index + 1,
                            text=text,
                            source_type=SourceType.NATIVE_TEXT,
                        )
                    )
        except DocumentExtractionError:
            raise
        except Exception as exc:
            raise DocumentExtractionError(
                f"Unexpected failure while parsing '{file_path}': {exc}"
            ) from exc
        finally:
            doc.close()

        n_ocr = sum(1 for p in pages if p.source_type == SourceType.OCR)
        logger.info(
            "Extracted %d pages from %s (%d native-text, %d flagged for OCR)",
            len(pages),
            path.name,
            len(pages) - n_ocr,
            n_ocr,
        )
        return pages

    def _extract_page(
        self, page: "pymupdf.Page", page_index: int
    ) -> Tuple[str, Optional[bytes]]:
        try:
            text = page.get_text("text").strip()
        except Exception as exc:
            raise DocumentExtractionError(
                f"Failed to extract text from page {page_index + 1}: {exc}"
            ) from exc

        if len(text) >= self._min_chars:
            return text, None

        try:
            matrix = pymupdf.Matrix(self._render_zoom, self._render_zoom)
            pixmap = page.get_pixmap(matrix=matrix)
            image_bytes = pixmap.tobytes("png")
        except Exception as exc:
            raise DocumentExtractionError(
                f"Failed to rasterise page {page_index + 1} for OCR fallback: {exc}"
            ) from exc

        return "", image_bytes
