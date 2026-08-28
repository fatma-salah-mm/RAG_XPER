"""
core/ingestion/ocr_engine.py for RAG_XPER

Dedicated OCR engine for image files and scanned PDF pages.
Supports EasyOCR and PaddleOCR with automatic paragraph line grouping.
"""
from __future__ import annotations

import io
from typing import List, Optional

import numpy as np
from PIL import Image

from core.exceptions import OCRExtractionError
from utils.logger import get_logger

logger = get_logger(__name__)


class OCREngine:
    """Extracts text from images / scanned document pages via OCR."""

    def __init__(self, engine: str = "easyocr", languages: Optional[List[str]] = None) -> None:
        self._engine_name = engine.lower()
        self._languages = languages or ["en", "ar"]
        self._reader = None
        logger.info(
            "OCREngine configured: engine=%s languages=%s", self._engine_name, self._languages
        )

    def _load_reader(self, engine_name: Optional[str] = None):
        target_engine = engine_name or self._engine_name
        if target_engine == self._engine_name and self._reader is not None:
            return self._reader

        try:
            if target_engine == "paddleocr":
                from paddleocr import PaddleOCR

                lang = self._languages[0] if self._languages else "en"
                try:
                    reader = PaddleOCR(lang=lang)
                except TypeError:
                    try:
                        reader = PaddleOCR(use_angle_cls=True, lang=lang)
                    except TypeError:
                        reader = PaddleOCR(lang=lang, show_log=False)
                if target_engine == self._engine_name:
                    self._reader = reader
                return reader

            elif target_engine == "easyocr":
                import easyocr

                reader = easyocr.Reader(self._languages, gpu=False)
                if target_engine == self._engine_name:
                    self._reader = reader
                return reader
            else:
                raise OCRExtractionError(f"Unsupported OCR engine: '{target_engine}'")
        except Exception as exc:
            if target_engine == "paddleocr" and self._engine_name == "paddleocr":
                logger.warning(
                    "PaddleOCR initialization failed (%s), falling back to EasyOCR", exc
                )
                try:
                    import easyocr

                    reader = easyocr.Reader(self._languages, gpu=False)
                    self._reader = reader
                    self._engine_name = "easyocr"
                    return reader
                except Exception as fallback_exc:
                    raise OCRExtractionError(
                        f"Failed to initialise both PaddleOCR ({exc}) and EasyOCR ({fallback_exc})"
                    ) from fallback_exc
            raise OCRExtractionError(
                f"Failed to initialise OCR engine '{target_engine}': {exc}"
            ) from exc

    def extract_text(self, image_bytes: bytes) -> str:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_array = np.array(image)
        except Exception as exc:
            raise OCRExtractionError(f"Invalid image data supplied to OCR: {exc}") from exc

        reader = self._load_reader()

        try:
            if self._engine_name == "paddleocr":
                try:
                    result = reader.ocr(image_array)
                except Exception:
                    result = reader.ocr(image_array, cls=True)
                lines = [
                    line[1][0]
                    for block in (result or [])
                    for line in (block or [])
                    if len(line) > 1 and isinstance(line[1], (list, tuple))
                ]
                text = "\n".join(lines)
            else:  # easyocr
                result = reader.readtext(image_array, paragraph=True)
                text = "\n".join(entry[1] for entry in result)
        except Exception as exc:
            logger.warning("OCR inference failed with %s: %s. Trying EasyOCR fallback...", self._engine_name, exc)
            try:
                import easyocr
                fallback_reader = easyocr.Reader(self._languages, gpu=False)
                result = fallback_reader.readtext(image_array, paragraph=True)
                text = "\n".join(entry[1] for entry in result)
            except Exception as fb_exc:
                raise OCRExtractionError(f"OCR inference failed: {exc} (fallback failed: {fb_exc})") from exc

        if not text.strip():
            logger.warning("OCR produced no text for this image (blank page or low-quality scan?).")

        return text.strip()

    def extract_from_file(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except OSError as exc:
            raise OCRExtractionError(f"Could not read image file '{image_path}': {exc}") from exc
        return self.extract_text(image_bytes)
