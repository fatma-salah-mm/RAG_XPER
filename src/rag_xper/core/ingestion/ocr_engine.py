"""
rag_xper.core.ingestion.ocr_engine

OCR engine supporting EasyOCR and PaddleOCR with line grouping for Arabic & English.
"""
from __future__ import annotations

import io
from typing import List

from PIL import Image

from rag_xper.core.exceptions import OCRExtractionError
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)


class OCREngine:
    """Extracts text from raw image bytes or image files using EasyOCR or PaddleOCR."""

    def __init__(self, engine: str = "easyocr", languages: List[str] | None = None) -> None:
        self._engine_type = engine.lower()
        self._languages = languages or ["en", "ar"]
        self._reader = None
        self._paddle = None

        if self._engine_type == "easyocr":
            try:
                import easyocr
                self._reader = easyocr.Reader(self._languages, gpu=False)
            except Exception as exc:
                logger.warning("Could not initialize EasyOCR: %s", exc)
        elif self._engine_type == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                self._paddle = PaddleOCR(use_angle_cls=True, lang="ar")
            except Exception as exc:
                logger.warning("Could not initialize PaddleOCR: %s", exc)

    def extract_text(self, image_bytes: bytes) -> str:
        """Run OCR on in-memory image bytes."""
        if not image_bytes:
            return ""

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise OCRExtractionError(f"Failed to decode image bytes: {exc}") from exc

        if self._engine_type == "easyocr":
            if self._reader is None:
                try:
                    import easyocr
                    self._reader = easyocr.Reader(self._languages, gpu=False)
                except Exception as exc:
                    raise OCRExtractionError(f"EasyOCR not available: {exc}") from exc

            try:
                import numpy as np
                img_array = np.array(image)
                results = self._reader.readtext(img_array, paragraph=True)
                lines = [res[1] for res in results if isinstance(res, (list, tuple)) and len(res) >= 2]
                return "\n".join(lines).strip()
            except Exception as exc:
                raise OCRExtractionError(f"EasyOCR failed: {exc}") from exc

        elif self._engine_type == "paddleocr":
            if self._paddle is None:
                try:
                    from paddleocr import PaddleOCR
                    self._paddle = PaddleOCR(use_angle_cls=True, lang="ar")
                except Exception as exc:
                    raise OCRExtractionError(f"PaddleOCR not available: {exc}") from exc

            try:
                import numpy as np
                img_array = np.array(image)
                result = self._paddle.ocr(img_array, cls=True)
                lines = []
                if result and result[0]:
                    for line in result[0]:
                        lines.append(line[1][0])
                return "\n".join(lines).strip()
            except Exception as exc:
                raise OCRExtractionError(f"PaddleOCR failed: {exc}") from exc

        raise OCRExtractionError(f"Unsupported OCR engine: '{self._engine_type}'")

    def extract_from_file(self, file_path: str) -> str:
        """Extract text from an image file on disk."""
        with open(file_path, "rb") as f:
            return self.extract_text(f.read())
