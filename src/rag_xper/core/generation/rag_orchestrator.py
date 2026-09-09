"""
rag_xper.core.generation.rag_orchestrator

High-level RAG Orchestrator coordinating Document Ingestion, Modular Chunking,
Deduplication via Document Identity (doc_id, file_hash, filename),
Hybrid Retrieval (BM25 + Qdrant RRF), and Chain-of-Thought Generation.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

from rag_xper.core.exceptions import DocumentExtractionError, OCRExtractionError
from rag_xper.core.generation.llm_interface import BaseLLM
from rag_xper.core.ingestion.document_extractor import DocumentExtractor
from rag_xper.core.ingestion.ocr_engine import OCREngine
from rag_xper.core.ingestion.text_chunker import ChunkerFactory, compute_content_hash
from rag_xper.core.models import Chunk, PageContent, RAGResponse, RetrievedChunk, SourceType
from rag_xper.core.retrieval.base_vector_store import BaseVectorStore
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}

# Every extension the pipeline can ingest, shared by the API, the CLI, and folder scans.
SUPPORTED_EXTENSIONS = {".pdf"} | _TEXT_EXTENSIONS | _IMAGE_EXTENSIONS

_COT_PROMPT_TEMPLATE = """You are a highly accurate, professional assistant answering questions strictly and faithfully based on the CONTEXT below.

CONTEXT:
{context}

QUESTION:
{question}

Instructions:
1. In a section titled "Reasoning:", think step by step in the SAME language as the question. Analyze ONLY the provided CONTEXT. Do NOT assume, speculate, or introduce external knowledge. Do NOT modify or rewrite names, numbers, or article identifiers.
2. In a section titled "Answer:", provide the final accurate answer clearly and concisely in the SAME language as the question, based strictly on the context. Every factual claim must cite the corresponding context block using [1], [2], etc.
3. If the context does not contain enough information to answer, state clearly that the requested information is not found in the documents. If conflicting evidence exists, explicitly quote the conflict.

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

    def ingest_file(
        self,
        file_path: str,
        strategy: Optional[str] = None,
        force: bool = False,
        original_filename: Optional[str] = None,
    ) -> int:
        """Ingest a file with stable document identity (doc_id, file_hash) and modular chunking."""
        path = Path(file_path)
        if not path.exists():
            raise DocumentExtractionError(f"File not found: {file_path}")

        # 1. Compute stable file_hash from file bytes
        try:
            with open(path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            file_hash = compute_content_hash(str(path))

        doc_name = original_filename or path.name
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_name}:{file_hash}"))

        if not force and self._vector_store.is_file_ingested(doc_name, content_hash=file_hash):
            logger.info("File '%s' (hash=%s) already indexed. Skipping re-embedding.", doc_name, file_hash[:8])
            return 0

        if force:
            self._vector_store.delete_file(doc_name)

        # 2. Extraction / OCR
        if path.suffix.lower() in _IMAGE_EXTENSIONS:
            logger.info("Routing image '%s' directly to OCREngine", doc_name)
            text = self._ocr.extract_from_file(str(path))
            pages = [
                PageContent(
                    source_path=doc_name,
                    page_number=1,
                    text=text,
                    source_type=SourceType.IMAGE_FILE,
                )
            ]
        else:
            pages = self._extractor.extract(str(path))
            pages = self._resolve_ocr_pages(pages)

        # Ensure consistent document naming on extracted pages
        for p in pages:
            p.source_path = doc_name

        # 3. Modular Chunking
        active_strategy = strategy or getattr(self._settings, "chunking_strategy", "auto")
        chunker = ChunkerFactory.create_chunker(active_strategy, self._settings)
        chunks = chunker.chunk_pages(pages)

        if not chunks:
            logger.warning("No text extracted from '%s'. Nothing to store.", doc_name)
            return 0

        # Attach stable document identity to each chunk
        for c in chunks:
            c.metadata["doc_id"] = doc_id
            c.metadata["filename"] = doc_name
            c.metadata["file_hash"] = file_hash
            c.metadata["source"] = doc_name

        # 4. Vector store persistence
        stored_count = self._vector_store.upsert_chunks(chunks)
        logger.info("Ingested '%s' -> %d chunks stored via strategy '%s'", doc_name, stored_count, active_strategy)
        return stored_count

    def ingest_directory(
        self,
        directory: str,
        strategy: Optional[str] = None,
        recursive: bool = False,
        force: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, object]:
        """Ingest every supported file in a folder."""
        root = Path(directory)
        if not root.is_dir():
            raise DocumentExtractionError(f"Directory not found: {directory}")

        pattern = "**/*" if recursive else "*"
        files = sorted(
            p for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        logger.info("Scanning '%s' -> %d supported file(s)", root, len(files))

        report: Dict[str, object] = {
            "directory": str(root),
            "files": [],
            "ingested": 0,
            "skipped": 0,
            "failed": 0,
            "total_chunks": 0,
        }

        for index, path in enumerate(files, start=1):
            try:
                chunks = self.ingest_file(str(path), strategy=strategy, force=force)
                outcome = "ingested" if chunks > 0 else "skipped"
                report[outcome] += 1
                report["total_chunks"] += chunks
                report["files"].append({"file": path.name, "status": outcome, "chunks": chunks})
            except Exception as exc:
                logger.error("Failed to ingest '%s': %s", path.name, exc)
                report["failed"] += 1
                report["files"].append(
                    {"file": path.name, "status": "failed", "chunks": 0, "error": str(exc)}
                )

            if progress_callback:
                progress_callback(index, len(files))

        logger.info(
            "Directory ingest complete: %d ingested, %d skipped, %d failed, %d chunks",
            report["ingested"], report["skipped"], report["failed"], report["total_chunks"],
        )
        return report

    def _resolve_ocr_pages(self, pages: List[PageContent]) -> List[PageContent]:
        resolved: List[PageContent] = []
        for page in pages:
            if page.source_type == SourceType.OCR:
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
        filter_filename: Optional[str] = None,
        filter_article: Optional[str] = None,
    ) -> RAGResponse:
        """Query knowledge base and generate faithful CoT response with strict citations."""
        is_arabic = any("\u0600" <= ch <= "\u06FF" for ch in question)
        empty_answer = (
            "لم يتم العثور على أي معلومات ذات صلة في المستندات المرفقة."
            if is_arabic
            else "No relevant information found in the provided documents."
        )
        empty_reasoning = (
            "لا توجد نصوص أو مصادر متطابقة في قاعدة البيانات."
            if is_arabic
            else "No matching passages or sources found in the database."
        )

        should_hybrid = (
            use_hybrid
            if use_hybrid is not None
            else getattr(self._settings, "use_hybrid_search", True)
        )

        # 1. Retrieval
        if should_hybrid:
            retrieved = self._vector_store.hybrid_search(
                query=question,
                top_k=min(top_k, 10),
                fetch_k=getattr(self._settings, "fetch_k", 25),
                alpha=getattr(self._settings, "hybrid_alpha", 0.5),
            )
        else:
            retrieved = self._vector_store.similarity_search(query=question, top_k=min(top_k, 10))

        # Apply explicit filters if provided (T2.5)
        if filter_filename:
            fn_norm = Path(filter_filename).name.lower()
            retrieved = [
                r for r in retrieved
                if r.chunk.metadata.get("filename", "").lower() == fn_norm
                or Path(r.chunk.metadata.get("source", "")).name.lower() == fn_norm
            ]

        if filter_article:
            retrieved = [
                r for r in retrieved
                if str(r.chunk.metadata.get("article_number", "")).strip() == str(filter_article).strip()
            ]

        # Apply minimum retrieval score threshold (T2.4)
        min_score = getattr(self._settings, "min_retrieval_score", 0.0) if self._settings else 0.0
        retrieved = [r for r in retrieved if r.score >= min_score]

        if not retrieved:
            return RAGResponse(
                answer=empty_answer,
                reasoning=empty_reasoning,
                sources=[],
                query=question,
            )

        # 2. Parent-Child Context Resolution & Deduplication
        context_blocks: List[str] = []
        seen_parents = set()
        resolved_sources: List[RetrievedChunk] = []

        for i, r in enumerate(retrieved, start=1):
            meta = r.chunk.metadata or {}
            source_name = meta.get("filename") or Path(meta.get("source", "doc")).name
            page_num = meta.get("page", 1)
            src_type = meta.get("source_type", "text")

            parent_id = meta.get("parent_id")
            if parent_id and meta.get("parent_text"):
                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)
                content_text = meta["parent_text"]
                # Align source text with LLM prompt context (T2.4)
                resolved_chunk = Chunk(
                    chunk_id=r.chunk.chunk_id,
                    text=content_text,
                    metadata=r.chunk.metadata,
                )
                resolved_sources.append(RetrievedChunk(chunk=resolved_chunk, score=r.score))
            else:
                content_text = r.chunk.text
                resolved_sources.append(r)

            block = f"[{i}] {source_name} (Page {page_num}, type={src_type}):\n{content_text}"
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks)

        # 3. Prompt Formatting & Generation
        prompt = _COT_PROMPT_TEMPLATE.format(context=context_str, question=question)
        raw_completion = self._llm.generate(prompt)

        # 4. Parsing Reasoning & Answer (Fail-closed on missing Answer: header)
        reasoning, answer = self._parse_cot_response(raw_completion, is_arabic=is_arabic)

        return RAGResponse(
            answer=answer,
            reasoning=reasoning,
            sources=resolved_sources,
            query=question,
        )

    def _parse_cot_response(self, raw_text: str, is_arabic: bool = True) -> tuple[Optional[str], str]:
        clean_text = raw_text.strip()
        pattern = r"(?:Reasoning|التحليل|التفكير):\s*(.*?)\s*(?:Answer|الإجابة|النتيجة):\s*(.*)"
        match = re.search(pattern, clean_text, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip(), match.group(2).strip()

        # Fail-closed if model omits Answer: header (T2.3)
        logger.warning("LLM response did not contain clear 'Answer:' header.")
        if "Reasoning:" in clean_text or "التحليل:" in clean_text:
            return clean_text, (
                "لم يتم استخلاص إجابة محددة وفق متطلبات التوثيق."
                if is_arabic
                else "Could not extract a verified answer meeting citation requirements."
            )

        return None, clean_text
