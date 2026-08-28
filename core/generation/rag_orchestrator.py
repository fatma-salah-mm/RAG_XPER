"""
core/generation/rag_orchestrator.py for RAG_XPER

High-level facade coordinating extraction, modular chunking, vector storage (Qdrant/ChromaDB),
hybrid retrieval, parent-child context resolution, and Chain-of-Thought generation.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from core.exceptions import DocumentExtractionError, OCRExtractionError
from core.generation.llm_interface import BaseLLM
from core.ingestion.document_extractor import DocumentExtractor
from core.ingestion.ocr_engine import OCREngine
from core.ingestion.text_chunker import ChunkerFactory
from core.models import Chunk, PageContent, RAGResponse, RetrievedChunk, SourceType
from core.retrieval.base_vector_store import BaseVectorStore
from utils.logger import get_logger

logger = get_logger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}

_COT_PROMPT_TEMPLATE = """You are a highly accurate, professional assistant answering questions strictly and faithfully based on the CONTEXT below.

CONTEXT:
{context}

QUESTION:
{question}

Instructions:
1. In a section titled "Reasoning:", think step by step in the SAME language as the question. Carefully analyze the context, verify exact numbers, article numbers, and dates. If the context originates from OCR (scanned pages), intelligently correct minor OCR typographical/character artifact errors (e.g. Arabic dot confusions such as 'فنحي' -> 'فتحي', 'جبمس' -> 'جيمس', or broken ligatures) to extract names and facts faithfully.
2. In a section titled "Answer:", provide the final accurate answer clearly and concisely in the SAME language as the question, based strictly on the context.
3. If the context truly does not contain enough information to answer, state that clearly in the Answer section instead of guessing.

Reasoning:
"""


class RAGOrchestrator:
    """High-level orchestrator for the RAG_XPER pipeline."""

    def __init__(
        self,
        extractor: DocumentExtractor,
        ocr: OCREngine,
        vector_store: BaseVectorStore,
        llm: BaseLLM,
        settings=None,
    ) -> None:
        self._extractor = extractor
        self._ocr = ocr
        self._vector_store = vector_store
        self._llm = llm
        self._settings = settings

    def ingest_file(self, file_path: str, strategy: Optional[str] = None) -> int:
        """Ingest a file with the chosen or configured chunking strategy."""
        path = Path(file_path)
        if not path.exists():
            raise DocumentExtractionError(f"File not found: {file_path}")

        # Check if already ingested in vector store to save API quota
        if self._vector_store.is_file_ingested(file_path):
            logger.info("File '%s' already indexed in vector database. Skipping re-embedding.", path.name)
            return 0

        # 1. Extraction / OCR
        if path.suffix.lower() in _IMAGE_EXTENSIONS:
            logger.info("Routing standalone image '%s' directly to OCREngine", path.name)
            text = self._ocr.extract_from_file(str(path))
            pages = [
                PageContent(
                    source_path=str(path),
                    page_number=1,
                    text=text,
                    source_type=SourceType.IMAGE_FILE,
                )
            ]
        else:
            pages = self._extractor.extract(str(path))
            pages = self._resolve_ocr_pages(pages)

        # 2. Modular Chunking
        active_strategy = strategy or getattr(self._settings, "chunking_strategy", "recursive")
        chunker = ChunkerFactory.create_chunker(active_strategy, self._settings)
        chunks = chunker.chunk_pages(pages)
        logger.info(
            "Chunked %d pages of '%s' into %d chunks using strategy '%s'",
            len(pages), path.name, len(chunks), active_strategy
        )

        if not chunks:
            logger.warning("No text extracted from '%s'. Nothing to store.", path.name)
            return 0

        # 3. Vector store persistence
        stored_count = self._vector_store.upsert_chunks(chunks)
        logger.info("Ingested '%s' -> %d chunks stored", path.name, stored_count)
        return stored_count

    def _resolve_ocr_pages(self, pages: List[PageContent]) -> List[PageContent]:
        resolved: List[PageContent] = []
        for page in pages:
            if page.source_type == SourceType.OCR and page.raw_image is not None:
                try:
                    ocr_text = self._ocr.extract_text(page.raw_image)
                    resolved.append(
                        PageContent(
                            source_path=page.source_path,
                            page_number=page.page_number,
                            text=ocr_text,
                            source_type=SourceType.OCR,
                        )
                    )
                except OCRExtractionError as exc:
                    logger.warning("OCR failed on page %d: %s. Continuing with empty text.", page.page_number, exc)
                    resolved.append(page)
            else:
                resolved.append(page)
        return resolved

    def query(
        self,
        question: str,
        top_k: int = 6,
        use_hybrid: Optional[bool] = None,
    ) -> RAGResponse:
        """Query the knowledge base and generate a faithful CoT response."""
        should_hybrid = (
            use_hybrid
            if use_hybrid is not None
            else getattr(self._settings, "use_hybrid_search", True)
        )

        # 1. Retrieval
        if should_hybrid:
            retrieved = self._vector_store.hybrid_search(
                query=question,
                top_k=top_k,
                fetch_k=getattr(self._settings, "fetch_k", 25),
                alpha=getattr(self._settings, "hybrid_alpha", 0.5),
            )
        else:
            retrieved = self._vector_store.similarity_search(query=question, top_k=top_k)

        if not retrieved:
            return RAGResponse(
                answer="لم يتم العثور على أي معلومات ذات صلة في المستندات المرفقة.",
                reasoning="لا توجد نصوص أو مصادر متطابقة في قاعدة البيانات.",
                sources=[],
                query=question,
            )

        # 2. Parent-Child Context Resolution & Deduplication
        context_blocks: List[str] = []
        seen_parents = set()

        for i, r in enumerate(retrieved, start=1):
            meta = r.chunk.metadata or {}
            source_name = Path(meta.get("source", "doc")).name
            page_num = meta.get("page", 1)
            src_type = meta.get("source_type", "text")

            # If this is a child chunk with parent_text, resolve to the full parent text
            parent_id = meta.get("parent_id")
            if parent_id and meta.get("parent_text"):
                if parent_id in seen_parents:
                    continue  # Deduplicate identical parents
                seen_parents.add(parent_id)
                content_text = meta["parent_text"]
            else:
                content_text = r.chunk.text

            block = (
                f"[{i}] {source_name} (Page {page_num}, type={src_type}):\n"
                f"{content_text}"
            )
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks)

        # 3. Prompt Formatting & LLM Generation
        prompt = _COT_PROMPT_TEMPLATE.format(context=context_str, question=question)
        raw_completion = self._llm.generate(prompt)

        # 4. Parsing Reasoning & Answer
        reasoning, answer = self._parse_cot_response(raw_completion)

        return RAGResponse(
            answer=answer,
            reasoning=reasoning,
            sources=retrieved,
            query=question,
        )

    def _parse_cot_response(self, raw_text: str) -> tuple[Optional[str], str]:
        """Split LLM response into (reasoning, answer)."""
        clean_text = raw_text.strip()

        # Check for explicit Reasoning / Answer sections
        pattern = r"(?:Reasoning|التحليل|التفكير):\s*(.*?)\s*(?:Answer|الإجابة|النتيجة):\s*(.*)"
        match = re.search(pattern, clean_text, re.DOTALL | re.IGNORECASE)

        if match:
            reasoning = match.group(1).strip()
            answer = match.group(2).strip()
            return reasoning, answer

        # Fallback if no delimiter
        return None, clean_text
