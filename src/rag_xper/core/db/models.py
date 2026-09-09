"""
rag_xper.core.db.models

SQLAlchemy Models for Books, Documents Catalog, and Query Logs.
"""
from __future__ import annotations

import datetime
from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Book(Base):
    """Represents a book or document registered in the system."""

    __tablename__ = "books"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False, index=True)
    author = Column(String(255), nullable=True)
    category = Column(String(100), nullable=True, default="General")
    filename = Column(String(255), nullable=False, unique=True, index=True)
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(BigInteger, default=0)
    content_hash = Column(String(64), nullable=True, index=True)
    total_pages = Column(Integer, default=1)
    chunk_count = Column(Integer, default=0)
    strategy_used = Column(String(50), default="recursive")
    status = Column(String(50), default="indexed")  # indexed, pending, failed
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "category": self.category,
            "filename": self.filename,
            "file_size_bytes": self.file_size_bytes,
            "total_pages": self.total_pages,
            "chunk_count": self.chunk_count,
            "strategy_used": self.strategy_used,
            "status": self.status,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class QueryLog(Base):
    """Log of user questions, answers, reasoning, and response times."""

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=True)
    sources_json = Column(Text, nullable=True)
    execution_time_ms = Column(Float, default=0.0)
    is_cached = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "question": self.question,
            "answer": self.answer,
            "reasoning": self.reasoning,
            "sources_json": self.sources_json,
            "execution_time_ms": self.execution_time_ms,
            "is_cached": self.is_cached,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
