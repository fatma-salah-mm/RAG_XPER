"""
core/ingestion/text_chunker.py for RAG_XPER

Modular, multi-strategy text chunking architecture.
Supports:
1. RecursiveChunker: General-purpose paragraph and sentence-aware chunking.
2. ParentChildChunker: Small-to-big hierarchical chunking for high retrieval precision.
3. ArticleBasedChunker: Structure-aware legal and article splitter.
4. AutoDetectChunker: Intelligently selects the best strategy based on document content.
"""
from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from core.exceptions import ChunkingError
from core.models import Chunk, PageContent
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk_pages(self, pages: List[PageContent]) -> List[Chunk]:
        """Split a list of extracted pages into chunks."""
        raise NotImplementedError


class RecursiveChunker(BaseChunker):
    """Standard recursive paragraph and sentence-aware sliding window chunker."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or _DEFAULT_SEPARATORS

    def chunk_pages(self, pages: List[PageContent]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for page in pages:
            if not page.text or not page.text.strip():
                continue
            try:
                pieces = self._split(page.text)
            except Exception as exc:
                raise ChunkingError(
                    f"Failed to chunk page {page.page_number} of {page.source_path}: {exc}"
                ) from exc

            for i, piece in enumerate(pieces):
                chunk_id = f"{page.source_path}:p{page.page_number}:c{i}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=piece,
                        metadata={
                            "source": page.source_path,
                            "page": page.page_number,
                            "chunk_index": i,
                            "source_type": page.source_type.value,
                            "strategy": "recursive",
                        },
                    )
                )
        return chunks

    def _split(self, text: str) -> List[str]:
        raw_pieces = self._split_recursive(text, self._separators)
        merged: List[str] = []
        current = ""

        for piece in raw_pieces:
            candidate = piece if not current else f"{current} {piece}".strip()
            if len(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                    overlap_seed = current[-self._chunk_overlap:] if self._chunk_overlap > 0 else ""
                    current = f"{overlap_seed} {piece}".strip() if overlap_seed else piece
                else:
                    merged.append(piece[:self._chunk_size])
                    current = piece[self._chunk_size:].strip()

        if current:
            merged.append(current)

        return [p.strip() for p in merged if p.strip()]

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return list(text)

        splits = text.split(sep)
        result: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) > self._chunk_size and remaining_seps:
                result.extend(self._split_recursive(s, remaining_seps))
            else:
                result.append(s)
        return result


class ParentChildChunker(BaseChunker):
    """
    Hierarchical Small-to-Big Chunker:
    Creates large Parent chunks for context and small Child chunks for dense search embedding.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        child_chunk_size: int = 300,
        child_overlap: int = 50,
    ) -> None:
        if child_chunk_size >= parent_chunk_size:
            raise ChunkingError("child_chunk_size must be smaller than parent_chunk_size")
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self._parent_splitter = RecursiveChunker(chunk_size=parent_chunk_size, chunk_overlap=100)
        self._child_splitter = RecursiveChunker(chunk_size=child_chunk_size, chunk_overlap=child_overlap)

    def chunk_pages(self, pages: List[PageContent]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for page in pages:
            if not page.text or not page.text.strip():
                continue

            # 1. Create parent chunks
            parent_pieces = self._parent_splitter._split(page.text)
            for p_idx, p_text in enumerate(parent_pieces):
                parent_id = f"{page.source_path}:p{page.page_number}:parent_{p_idx}"

                # 2. Split each parent into small child chunks for retrieval
                child_pieces = self._child_splitter._split(p_text)
                if not child_pieces:
                    child_pieces = [p_text]

                for c_idx, c_text in enumerate(child_pieces):
                    child_id = f"{parent_id}:child_{c_idx}"
                    chunks.append(
                        Chunk(
                            chunk_id=child_id,
                            text=c_text,
                            metadata={
                                "source": page.source_path,
                                "page": page.page_number,
                                "parent_id": parent_id,
                                "parent_text": p_text,
                                "child_index": c_idx,
                                "is_child": True,
                                "source_type": page.source_type.value,
                                "strategy": "parent_child",
                            },
                        )
                    )
        return chunks


class ArticleBasedChunker(BaseChunker):
    """
    Structure-aware legal chunker that splits on Arabic and English legal article headers.
    """

    ARTICLE_REGEX = re.compile(
        r"(?=(?:^|\n)(?:المادة|مادة|البند|فصل|الفصل|Article|Section|Chapter)\s+(?:[^\n:]{1,40})[:\n\.-])",
        re.IGNORECASE,
    )

    def __init__(self, fallback_max_size: int = 1500) -> None:
        self.fallback_max_size = fallback_max_size
        self._fallback = RecursiveChunker(chunk_size=fallback_max_size, chunk_overlap=150)

    def chunk_pages(self, pages: List[PageContent]) -> List[Chunk]:
        chunks: List[Chunk] = []

        for page in pages:
            if not page.text or not page.text.strip():
                continue

            # Split on article headings
            sections = self.ARTICLE_REGEX.split(page.text)
            clean_sections = [s.strip() for s in sections if s and s.strip()]

            if len(clean_sections) <= 1:
                # If no article headers found on this page, use recursive fallback
                page_chunks = self._fallback.chunk_pages([page])
                for pc in page_chunks:
                    pc.metadata["strategy"] = "article_based (fallback)"
                chunks.extend(page_chunks)
            else:
                for i, sec in enumerate(clean_sections):
                    if len(sec) > self.fallback_max_size:
                        # If an article is extraordinarily long, sub-split it
                        sub_pieces = self._fallback._split(sec)
                        for j, sub in enumerate(sub_pieces):
                            chunk_id = f"{page.source_path}:p{page.page_number}:art_{i}_sub_{j}"
                            chunks.append(
                                Chunk(
                                    chunk_id=chunk_id,
                                    text=sub,
                                    metadata={
                                        "source": page.source_path,
                                        "page": page.page_number,
                                        "article_index": i,
                                        "source_type": page.source_type.value,
                                        "strategy": "article_based",
                                    },
                                )
                            )
                    else:
                        chunk_id = f"{page.source_path}:p{page.page_number}:art_{i}"
                        chunks.append(
                            Chunk(
                                chunk_id=chunk_id,
                                text=sec,
                                metadata={
                                    "source": page.source_path,
                                    "page": page.page_number,
                                    "article_index": i,
                                    "source_type": page.source_type.value,
                                    "strategy": "article_based",
                                },
                            )
                        )
        return chunks


class AutoDetectChunker(BaseChunker):
    """
    Intelligently analyzes the document content and selects the optimal chunking strategy:
    - ArticleBasedChunker for legal codes, contracts, and regulations.
    - RecursiveChunker for general books, documents, and scanned pages.
    """

    def __init__(self, recursive_chunker: RecursiveChunker, article_chunker: ArticleBasedChunker) -> None:
        self.recursive = recursive_chunker
        self.article = article_chunker

    def chunk_pages(self, pages: List[PageContent]) -> List[Chunk]:
        sample_text = " ".join([p.text for p in pages[:5] if p.text])
        is_legal = bool(re.search(r"(المادة\s+|نظام\s+|لائحة\s+|Article\s+\d+|مرسوم\s+ملكي)", sample_text, re.IGNORECASE))

        if is_legal:
            logger.info("AutoDetect: Legal/Article structure detected -> Using ArticleBasedChunker")
            return self.article.chunk_pages(pages)
        else:
            logger.info("AutoDetect: General document structure detected -> Using RecursiveChunker")
            return self.recursive.chunk_pages(pages)


class ChunkerFactory:
    """Factory to instantiate the appropriate Chunker based on strategy name."""

    @staticmethod
    def create_chunker(strategy: str = "recursive", config=None) -> BaseChunker:
        strat = strategy.lower().strip()
        chunk_size = getattr(config, "chunk_size", 1000) if config else 1000
        chunk_overlap = getattr(config, "chunk_overlap", 150) if config else 150
        parent_size = getattr(config, "parent_chunk_size", 1500) if config else 1500
        child_size = getattr(config, "child_chunk_size", 300) if config else 300

        if "parent" in strat or "child" in strat:
            return ParentChildChunker(parent_chunk_size=parent_size, child_chunk_size=child_size)
        elif "art" in strat or "legal" in strat or "قانون" in strat or "ماد" in strat:
            return ArticleBasedChunker()
        elif "auto" in strat or "تلقائي" in strat:
            rec = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            art = ArticleBasedChunker()
            return AutoDetectChunker(rec, art)
        else:
            return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
