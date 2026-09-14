"""FastAPI dependencies for authentication and shared resources."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from rag_xper.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _auth_required() -> bool:
    return settings.require_auth or bool(settings.api_keys)


def verify_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    """Validate API key using constant-time comparison, failing closed if auth required."""
    if not _auth_required():
        return api_key

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key in 'X-API-Key' header.",
        )

    is_valid = any(secrets.compare_digest(api_key, valid_key) for valid_key in settings.api_keys)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key in 'X-API-Key' header.",
        )
    return api_key


def verify_metrics_access(api_key: str | None = Security(api_key_header)) -> None:
    """Protect /metrics when METRICS_REQUIRE_AUTH is enabled."""
    if settings.metrics_require_auth:
        verify_api_key(api_key)
