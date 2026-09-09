"""
rag_xper.core.db package

Database ORM and catalog persistence for books, documents, and query history.
"""
from rag_xper.core.db.models import Base, Book, QueryLog
from rag_xper.core.db.session import get_db, init_db

__all__ = ["Base", "Book", "QueryLog", "get_db", "init_db"]
