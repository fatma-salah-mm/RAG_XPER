"""
rag_xper.utils.logger

Unified logging configuration for RAG_XPER supporting standard text and structured JSON formats.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


class RequestIdFilter(logging.Filter):
    """Attach the current request_id context variable to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get("")
        return True


class JSONFormatter(logging.Formatter):
    """Structured JSON formatter for production log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id") and record.request_id:
            log_obj["request_id"] = record.request_id
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    use_json = os.getenv("LOG_FORMAT", "text").lower() == "json"
    handler = logging.StreamHandler(sys.stdout)

    if use_json:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler.addFilter(RequestIdFilter())
        root.addHandler(handler)

    _configured = True


def set_request_id(request_id: str) -> contextvars.Token:
    """Bind request_id for the current async context."""
    return request_id_ctx.set(request_id)


def reset_request_id(token: contextvars.Token) -> None:
    """Restore the previous request_id after a request completes."""
    request_id_ctx.reset(token)


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with the standard RAG_XPER format."""
    _configure_root()
    return logging.getLogger(name)
