"""
main.py for RAG_XPER

Command-line entrypoint for the RAG_XPER pipeline.
Features:
- Native Windows UTF-8 console output support
- Interactive native file picker dialog
- Interactive Chunking Strategy selection (Recursive, Parent-Child, Article-Based, Auto-Detect)
- Qdrant Vector Store (Rust-powered local embedded) + ChromaDB support
- Interactive Q&A chat loop with Chain-of-Thought reasoning & source attribution
"""
from __future__ import annotations

import argparse
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Optional

# Reconfigure Windows standard output streams to UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import settings
from core.generation import GeminiLLM, OllamaLLM, RAGOrchestrator
from core.ingestion import DocumentExtractor, OCREngine
from core.retrieval import VectorStoreFactory
from utils.logger import get_logger

logger = get_logger(__name__)


def pick_file_dialog() -> Optional[str]:
    """Open a native Windows file dialog to pick a PDF or image file."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()

    file_path = filedialog.askopenfilename(
        title="RAG_XPER - اختاري ملف PDF أو صورة للمعالجة",
        filetypes=[
            ("Supported Files", "*.pdf;*.png;*.jpg;*.jpeg;*.tiff;*.bmp;*.webp"),
            ("PDF Documents", "*.pdf"),
            ("Images", "*.png;*.jpg;*.jpeg;*.tiff;*.bmp;*.webp"),
            ("All Files", "*.*"),
        ],
    )
    root.destroy()
    return file_path if file_path else None


def choose_strategy_interactive() -> str:
    """Prompt the user in CLI to pick a chunking strategy."""
    print("\n" + "=" * 55)
    print("📋 اختاري استراتيجية التقطيع (Chunking Strategy):")
    print("  [1] Recursive (عام لجميع الكتب والمستندات - الافتراضي)")
    print("  [2] Parent-Child (تقطيع هرمي دقيق جداً للمستندات الكبيرة)")
    print("  [3] Article-Based (مخصص للمواد واللوائح القانونية والعقود)")
    print("  [4] Auto-Detect (فحص تلقائي ذكي لنوع المستند)")
    print("=" * 55)

    choice = input("👉 اختيارك [1/2/3/4] (اضغطي Enter للافتراضي): ").strip()
    if choice == "2":
        return "parent_child"
    elif choice == "3":
        return "article_based"
    elif choice == "4":
        return "auto"
    else:
        return "recursive"


def build_orchestrator() -> RAGOrchestrator:
    """Instantiate and wire all pipeline components together."""
    settings.validate()

    # 1. LLM Client
    if settings.llm_provider == "gemini":
        llm = GeminiLLM(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model,
            embedding_model=settings.gemini_embedding_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    else:
        llm = OllamaLLM(
            base_url=settings.ollama_base_url,
            model_name=settings.ollama_model,
            embedding_model=settings.ollama_embedding_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    # 2. Vector Store (Qdrant by default or ChromaDB)
    vector_store = VectorStoreFactory.create_vector_store(
        config=settings,
        embedding_fn=llm.embed,
    )

    # 3. Document Extractor & OCR Engine
    extractor = DocumentExtractor(
        native_text_min_chars=settings.native_text_min_chars,
        render_zoom=settings.ocr_render_zoom,
    )
    ocr = OCREngine(
        engine=settings.ocr_engine,
        languages=list(settings.ocr_languages),
    )

    return RAGOrchestrator(
        extractor=extractor,
        ocr=ocr,
        vector_store=vector_store,
        llm=llm,
        settings=settings,
    )


def interactive_session() -> None:
    """Run an interactive CLI session with file selection, chunking choice, and Q&A."""
    orchestrator = build_orchestrator()

    print("\n" + "=" * 60)
    print("🚀 مرحباً بك في RAG_XPER (Enterprise Qdrant + Modular RAG)")
    print(f"🗄️  قاعدة المتجهات المفعلة: {settings.vector_store_type.upper()}")
    print("=" * 60)

    print("\nجاري فتح نافذة اختيار الملف...")
    file_path = pick_file_dialog()

    if file_path:
        print(f"\n📄 تم اختيار الملف: {Path(file_path).name}")
        strategy = choose_strategy_interactive()
        print(f"\n⏳ جاري فحص ومعالجة الملف باستخدام استراتيجية: [{strategy}] ...")
        n_chunks = orchestrator.ingest_file(file_path, strategy=strategy)
        if n_chunks > 0:
            print(f"✅ تمت معالجة وفهرسة {n_chunks} قطعة نصية بنجاح!")
        else:
            print("ℹ️  الملف مفهرس مسبقاً أو لا يتطلب معالجة جديدة.")
    else:
        print("ℹ️  لم يتم اختيار ملف جديد. سيتم استخدام المستندات المفهرسة مسبقاً في قاعدة البيانات.")

    print("\n" + "-" * 60)
    print("💬 يمكنك الآن طرح أسئلتك حول المستند (اكتبي 'exit' أو 'خروج' للإنهاء):")
    print("-" * 60)

    while True:
        try:
            question = input("\n❓ سؤالك: ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q", "خروج", "انهاء"):
                print("\n👋 إلى اللقاء!")
                break

            response = orchestrator.query(question, top_k=settings.top_k)

            if response.reasoning:
                print("\n--- 🧠 التحليل المنطقي (Reasoning) ---")
                print(response.reasoning)

            print("\n--- 💡 الإجابة (Answer) ---")
            print(response.answer)

            if response.sources:
                print("\n--- 📚 المصادر المستند إليها ---")
                for i, src in enumerate(response.sources, 1):
                    src_name = Path(src.chunk.metadata.get("source", "doc")).name
                    page = src.chunk.metadata.get("page", 1)
                    stype = src.chunk.metadata.get("source_type", "text")
                    strat = src.chunk.metadata.get("strategy", "default")
                    print(f"[{i}] {src_name} (صفحة {page}, نوع={stype}, استراتيجية={strat}), score={src.score:.3f}")

        except KeyboardInterrupt:
            print("\n\n👋 تم إنهاء الجلسة.")
            break
        except Exception as exc:
            logger.error("Error during query: %s", exc)
            print(f"❌ حدث خطأ: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG_XPER CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Ingest command
    ingest_p = subparsers.add_parser("ingest", help="Ingest a document into the vector store")
    ingest_p.add_argument("file_path", help="Path to PDF or image file")
    ingest_p.add_argument("--strategy", choices=["recursive", "parent_child", "article_based", "auto"], default=None)

    # Ask command
    ask_p = subparsers.add_parser("ask", help="Query the RAG pipeline")
    ask_p.add_argument("question", help="Question to ask")
    ask_p.add_argument("--top-k", type=int, default=6)

    args = parser.parse_args()

    if args.command == "ingest":
        orchestrator = build_orchestrator()
        n = orchestrator.ingest_file(args.file_path, strategy=args.strategy)
        print(f"Ingested {n} chunks from '{args.file_path}'")
    elif args.command == "ask":
        orchestrator = build_orchestrator()
        resp = orchestrator.query(args.question, top_k=args.top_k)
        if resp.reasoning:
            print(f"\n--- Reasoning ---\n{resp.reasoning}")
        print(f"\n--- Answer ---\n{resp.answer}")
    else:
        interactive_session()


if __name__ == "__main__":
    main()
