"""
rag_xper.core.db.service

Service layer for Books catalog and Query logging operations.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from rag_xper.core.db.models import Book, QueryLog
from rag_xper.core.db.session import get_db_session
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)


def register_book(
    title: str,
    filename: str,
    file_path: str,
    author: Optional[str] = None,
    category: Optional[str] = "General",
    file_size_bytes: int = 0,
    content_hash: Optional[str] = None,
    total_pages: int = 1,
    chunk_count: int = 0,
    strategy_used: str = "recursive",
    notes: Optional[str] = None,
) -> Book:
    """Insert or update a book/document in the catalog."""
    with get_db_session() as session:
        book = session.query(Book).filter(Book.filename == filename).first()
        if not book:
            book = Book(
                title=title,
                author=author,
                category=category,
                filename=filename,
                file_path=file_path,
                file_size_bytes=file_size_bytes,
                content_hash=content_hash,
                total_pages=total_pages,
                chunk_count=chunk_count,
                strategy_used=strategy_used,
                status="indexed",
                notes=notes,
            )
            session.add(book)
        else:
            book.title = title
            book.author = author or book.author
            book.category = category or book.category
            book.chunk_count = chunk_count or book.chunk_count
            book.status = "indexed"
            book.notes = notes or book.notes

        session.flush()
        logger.info("Book '%s' registered in database (ID: %s)", title, book.id)
        return book


def list_books(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch all cataloged books with optional category filter."""
    with get_db_session() as session:
        query = session.query(Book)
        if category:
            query = query.filter(Book.category == category)
        books = query.order_by(Book.created_at.desc()).all()
        return [b.to_dict() for b in books]


def delete_book(filename: str) -> bool:
    """Remove a book record from database."""
    with get_db_session() as session:
        book = session.query(Book).filter(Book.filename == filename).first()
        if book:
            session.delete(book)
            logger.info("Book '%s' deleted from database.", filename)
            return True
        return False


def log_query(
    question: str,
    answer: str,
    reasoning: Optional[str] = None,
    sources: Optional[List[Any]] = None,
    execution_time_ms: float = 0.0,
    is_cached: bool = False,
    session_id: Optional[str] = None,
) -> None:
    """Record query and response telemetry in database."""
    try:
        sources_str = json.dumps(sources, ensure_ascii=False) if sources else None
        with get_db_session() as session:
            log_entry = QueryLog(
                session_id=session_id,
                question=question,
                answer=answer,
                reasoning=reasoning,
                sources_json=sources_str,
                execution_time_ms=execution_time_ms,
                is_cached=is_cached,
            )
            session.add(log_entry)
    except Exception as exc:
        logger.warning("Failed to log query to database: %s", exc)


def get_query_history(limit: int = 50, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve recent query history."""
    try:
        with get_db_session() as session:
            query = session.query(QueryLog)
            if session_id:
                query = query.filter(QueryLog.session_id == session_id)
            logs = query.order_by(QueryLog.created_at.desc()).limit(limit).all()
            return [l.to_dict() for l in logs]
    except Exception as exc:
        logger.warning("Failed to fetch query history: %s", exc)
        return []
