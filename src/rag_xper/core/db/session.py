"""
rag_xper.core.db.session

Database Engine & Session Management supporting MySQL and local SQLite fallback.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from rag_xper.config import settings
from rag_xper.core.db.models import Base
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)

_engine = None
_SessionFactory = None


def get_database_url() -> str:
    """Build database connection URL based on settings (MySQL or fallback SQLite)."""
    if settings.mysql_host and settings.mysql_database:
        user = settings.mysql_user or "root"
        password = settings.mysql_password or ""
        host = settings.mysql_host
        port = settings.mysql_port or 3306
        db = settings.mysql_database
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8mb4"

    # Default fallback to SQLite in storage directory
    os.makedirs("./storage", exist_ok=True)
    return "sqlite:///./storage/rag_xper.db"


def init_engine():
    """Initialize SQLAlchemy engine and ensure schema is up to date."""
    global _engine, _SessionFactory
    if _engine is not None:
        return _engine

    db_url = get_database_url()
    mysql_configured = bool(settings.mysql_host and settings.mysql_database)
    try:
        if db_url.startswith("sqlite"):
            _engine = create_engine(db_url, connect_args={"check_same_thread": False})
        else:
            _engine = create_engine(db_url, pool_pre_ping=True, pool_size=5, max_overflow=10)

        if settings.run_db_migrations:
            from rag_xper.core.db.migrate import run_migrations

            run_migrations()
        else:
            Base.metadata.create_all(bind=_engine)

        _SessionFactory = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=_engine)
        logger.info("Database initialized successfully: %s", db_url.split("@")[-1] if "@" in db_url else db_url)
    except Exception as exc:
        if settings.mysql_required and mysql_configured:
            raise RuntimeError(f"MySQL is required but connection failed: {exc}") from exc
        logger.warning("Primary database connection failed (%s). Falling back to SQLite.", exc)
        _engine = create_engine("sqlite:///./storage/rag_xper.db", connect_args={"check_same_thread": False})
        if settings.run_db_migrations:
            from rag_xper.core.db.migrate import run_migrations

            run_migrations()
        else:
            Base.metadata.create_all(bind=_engine)
        _SessionFactory = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=_engine)

    return _engine


def init_db() -> None:
    """Explicit database schema initialization."""
    init_engine()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Provide a transactional database session."""
    if _SessionFactory is None:
        init_engine()
    session: Session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database session."""
    if _SessionFactory is None:
        init_engine()
    session: Session = _SessionFactory()
    try:
        yield session
    finally:
        session.close()
