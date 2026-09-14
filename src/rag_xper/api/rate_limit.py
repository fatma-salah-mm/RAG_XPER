"""Per-key / per-IP rate limiting for the REST API."""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from rag_xper.config import settings


def _rate_limit_key(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"
    return f"ip:{get_remote_address(request)}"


def _ask_rate_limit() -> str:
    rpm = settings.rate_limit_per_minute
    if rpm <= 0:
        return "10000/minute"
    return f"{rpm}/minute"


limiter = Limiter(key_func=_rate_limit_key, default_limits=[])
