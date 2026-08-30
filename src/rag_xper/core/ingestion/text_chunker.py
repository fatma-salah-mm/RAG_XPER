"""
rag_xper.core.ingestion.text_chunker

Modular Chunking Engine implementing the Strategy Pattern:
1. RecursiveChunker: General-purpose sliding window (default)
2. ParentChildChunker: Small-to-big hierarchical chunks
3. ArticleBasedChunker: Splits strictly by legal articles / sections
4. AutoDetectChunker: Inspects document heuristics to select optimal strategy
5. ChunkerFactory: Factory method to instantiate strategies
"""
from __future__ import annotations

import abc
import hashlib
import re
from typing import List, Optional

from rag_xper.core.exceptions import ChunkingError
from rag_xper.core.models import Chunk, PageContent
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of text for reliable content deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class BaseChunker(abc.ABC):
    """Abstract Base Strategy for all chunking implementations."""

    @abc.abstractmethod
    def chunk_pages(self, pages: List[PageContent]) -> List[Chunk]:
        """Given a list of extracted pages, return a list of Chunk objects."""
        raise NotImplementedError


class RecursiveChunker(BaseChunker):
    """General-purpose sliding-window chunker with natural delimiter fallback."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        self._chunk_size = chunk_size
        self._overlap = chunk_overlap

    def chunk_pages(self, pages: List[PageContent]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for page in pages:
            text = page.text.strip()
            if not text:
                continue

            page_chunks = self._split_text(text)
            for idx, c_text in enumerate(page_chunks):
                chunk_id = f"{page.source_path}:p{page.page_number}:c{idx}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=c_text,
                        metadata={
                            "source": page.source_path,
                            "page": page.page_number,
                            "source_type": page.source_type.value,
                            "chunk_index": idx,
                            "strategy": "recursive",
                            "content_hash": compute_content_hash(c_text),
                        },
                    )
                )
        return chunks

    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self._chunk_size:
            return [text]

        delimiters = ["\n\n", "\n", ". ", "، ", " ", ""]
        return self._recursive_split(text, delimiters, self._chunk_size, self._overlap)

    def _recursive_split(self, text: str, delimiters: List[str], chunk_size: int, overlap: int) -> List[str]:
        if not delimiters or len(text) <= chunk_size:
            return [text] if text else []

        delim = delimiters[0]
        sub_delims = delimiters[1:]

        parts = text.split(delim) if delim != "" else list(text)
        result: List[str] = []
        current: List[str] = []
        curr_len = 0

        for part in parts:
            part_len = len(part) + (len(delim) if delim != "" else 0)
            if curr_len + part_len > chunk_size and current:
                joined = delim.join(current) if delim != "" else "".join(current)
                if len(joined) > chunk_size and sub_delims:
                    result.extend(self._recursive_split(joined, sub_delims, chunk_size, overlap))
                else:
                    result.append(joined)

                # Keep overlap from the end
                overlap_accum: List[str] = []
                overlap_len = 0
                for prev in reversed(current):
                    if overlap_len + len(prev) <= overlap:
                        overlap_accum.insert(0, prev)
                        overlap_len += len(prev) + len(delim)
                    else:
                        break
                current = overlap_accum
                curr_len = overlap_len

            current.append(part)
            curr_len += part_len

        if current:
            joined = delim.join(current) if delim != "" else "".join(current)
            if len(joined) > chunk_size and sub_delims:
                result.extend(self._recursive_split(joined, sub_delims, chunk_size, overlap))
            else:
                result.append(joined)

        return [c.strip() for c in result if c.strip()]


class ParentChildChunker(BaseChunker):
    """Hierarchical Small-to-Big Chunker: Child chunks carry parent metadata."""

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        child_chunk_size: int = 300,
        child_overlap: int = 50,
    ) -> None:
        self._parent_chunk_size = parent_chunk_size
        self._child_chunk_size = child_chunk_size
        self._child_overlap = child_overlap
        self._parent_splitter = RecursiveChunker(chunk_size=parent_chunk_size, chunk_overlap=200)
        self._child_splitter = RecursiveChunker(chunk_size=child_chunk_size, chunk_overlap=child_overlap)

    def chunk_pages(self, pages: List[PageContent]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for page in pages:
            text = page.text.strip()
            if not text:
                continue

            parents = self._parent_splitter._split_text(text)
            for p_idx, parent_text in enumerate(parents):
                parent_id = f"{page.source_path}:p{page.page_number}:P{p_idx}"
                children = self._child_splitter._split_text(parent_text)

                for c_idx, child_text in enumerate(children):
                    child_id = f"{parent_id}:C{c_idx}"
                    all_chunks.append(
                        Chunk(
                            chunk_id=child_id,
                            text=child_text,
                            metadata={
                                "source": page.source_path,
                                "page": page.page_number,
                                "source_type": page.source_type.value,
                                "strategy": "parent_child",
                                "is_child": True,
                                "parent_id": parent_id,
                                "parent_text": parent_text,
                                "content_hash": compute_content_hash(child_text),
                            },
                        )
                    )
        return all_chunks


class ArticleBasedChunker(BaseChunker):
    """Legal & Regulatory Chunker: Splits text on article/clause boundaries."""

    _ARTICLE_PATTERN = re.compile(
        r"(?:^|\n)(?=(?:المادة\s+(?:[0-9]+|[\u0621-\u064A]+)|البند\s+(?:[0-9]+|[\u0621-\u064A]+)|الفصل\s+(?:[0-9]+|[\u0621-\u064A]+)|Article\s+[0-9]+|Section\s+[0-9]+|Clause\s+[0-9]+))",
        re.IGNORECASE | re.UNICODE,
    )

    def __init__(self, fallback_max_size: int = 1500) -> None:
        self._fallback_max_size = fallback_max_size
        self._fallback_chunker = RecursiveChunker(chunk_size=fallback_max_size, chunk_overlap=150)

    def chunk_pages(self, pages: List[PageContent]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for page in pages:
            text = page.text.strip()
            if not text:
                continue

            articles = self._ARTICLE_PATTERN.split(text)
            clean_articles = [a.strip() for a in articles if a.strip()]

            if not clean_articles:
                clean_articles = [text]

            for idx, art_text in enumerate(clean_articles):
                if len(art_text) > self._fallback_max_size:
                    sub_chunks = self._fallback_chunker._split_text(art_text)
                    for s_idx, s_text in enumerate(sub_chunks):
                        chunks.append(
                            Chunk(
                                chunk_id=f"{page.source_path}:p{page.page_number}:art{idx}:sub{s_idx}",
                                text=s_text,
                                metadata={
                                    "source": page.source_path,
                                    "page": page.page_number,
                                    "source_type": page.source_type.value,
                                    "strategy": "article_based",
                                    "content_hash": compute_content_hash(s_text),
                                },
                            )
                        )
                else:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{page.source_path}:p{page.page_number}:art{idx}",
                            text=art_text,
                            metadata={
                                "source": page.source_path,
                                "page": page.page_number,
                                "source_type": page.source_type.value,
                                "strategy": "article_based",
                                "content_hash": compute_content_hash(art_text),
                            },
                        )
                    )
        return chunks


class AutoDetectChunker(BaseChunker):
    """Automatically selects the best chunking strategy based on document content."""

    _LEGAL_KEYWORDS = ["المادة", "البند", "اللائحة", "المرسوم", "العقد", "نظام", "article", "clause", "agreement"]

    def __init__(self, settings=None) -> None:
        self._settings = settings

    def chunk_pages(self, pages: List[PageContent]) -> List[Chunk]:
        sample_text = " ".join([p.text[:500] for p in pages[:5]]).lower()
        is_legal = any(kw in sample_text for kw in self._LEGAL_KEYWORDS)

        if is_legal:
            logger.info("AutoDetect: Legal/Article-based document detected.")
            chunker = ArticleBasedChunker()
        else:
            logger.info("AutoDetect: General document detected. Using Recursive chunker.")
            c_size = getattr(self._settings, "chunk_size", 1000) if self._settings else 1000
            c_overlap = getattr(self._settings, "chunk_overlap", 150) if self._settings else 150
            chunker = RecursiveChunker(chunk_size=c_size, chunk_overlap=c_overlap)

        return chunker.chunk_pages(pages)


class ChunkerFactory:
    """Factory to instantiate the appropriate Chunker based on strategy key."""

    @staticmethod
    def create_chunker(strategy: str = "recursive", settings=None) -> BaseChunker:
        strat = (strategy or "recursive").lower()

        if strat == "parent_child":
            p_size = getattr(settings, "parent_chunk_size", 1500) if settings else 1500
            c_size = getattr(settings, "child_chunk_size", 300) if settings else 300
            return ParentChildChunker(parent_chunk_size=p_size, child_chunk_size=c_size)
        elif strat == "article_based":
            return ArticleBasedChunker()
        elif strat == "auto":
            return AutoDetectChunker(settings=settings)
        elif strat == "recursive":
            c_size = getattr(settings, "chunk_size", 1000) if settings else 1000
            c_overlap = getattr(settings, "chunk_overlap", 150) if settings else 150
            return RecursiveChunker(chunk_size=c_size, chunk_overlap=c_overlap)
        else:
            logger.warning("Unknown chunking strategy '%s'. Falling back to 'recursive'.", strategy)
            return RecursiveChunker()
