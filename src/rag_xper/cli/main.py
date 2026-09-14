"""
rag_xper.cli.main

Command-line entrypoint for RAG_XPER.
"""

from __future__ import annotations

import argparse
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rag_xper.bootstrap import build_orchestrator
from rag_xper.cli.interactive import run_interactive_session
from rag_xper.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG_XPER CLI — ingest documents and query the knowledge base.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "interactive",
        help="Launch an interactive session with file picker (desktop/demo)",
    )

    ingest_p = subparsers.add_parser("ingest", help="Ingest a document into the vector store")
    ingest_p.add_argument("file_path", help="Path to PDF, MD, TXT, or image file")
    ingest_p.add_argument(
        "--strategy",
        choices=["recursive", "parent_child", "article_based", "auto"],
        default=None,
    )
    ingest_p.add_argument("--force", action="store_true", help="Force re-indexing even if already indexed")

    dir_p = subparsers.add_parser("ingest-dir", help="Ingest every supported file in a folder")
    dir_p.add_argument(
        "directory",
        nargs="?",
        default=None,
        help=f"Folder to scan (defaults to DOCUMENTS_DIR='{settings.documents_dir}')",
    )
    dir_p.add_argument(
        "--strategy",
        choices=["recursive", "parent_child", "article_based", "auto"],
        default=None,
    )
    dir_p.add_argument("--recursive", action="store_true", help="Include sub-folders")
    dir_p.add_argument("--force", action="store_true", help="Force re-indexing even if already indexed")

    ask_p = subparsers.add_parser("ask", help="Query the RAG pipeline")
    ask_p.add_argument("question", help="Question to ask")
    ask_p.add_argument("--top-k", type=int, default=6)

    args = parser.parse_args()

    if args.command == "interactive":
        run_interactive_session()
    elif args.command == "ingest":
        orchestrator = build_orchestrator()
        n = orchestrator.ingest_file(args.file_path, strategy=args.strategy, force=args.force)
        print(f"Ingested {n} chunks from '{args.file_path}'")
    elif args.command == "ingest-dir":
        orchestrator = build_orchestrator()
        report = orchestrator.ingest_directory(
            args.directory or settings.documents_dir,
            strategy=args.strategy,
            recursive=args.recursive,
            force=args.force,
        )
        for item in report["files"]:
            print(f"[{item['status']:<8}] {item['file']} -> {item['chunks']} chunks")
        print(
            f"\nIngested {report['ingested']} file(s), skipped {report['skipped']}, "
            f"failed {report['failed']} — {report['total_chunks']} chunks total."
        )
    elif args.command == "ask":
        orchestrator = build_orchestrator()
        resp = orchestrator.query(args.question, top_k=args.top_k)
        if resp.reasoning:
            print(f"\n--- Reasoning ---\n{resp.reasoning}")
        print(f"\n--- Answer ---\n{resp.answer}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
