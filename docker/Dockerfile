# Multi-stage Dockerfile for RAG_XPER Enterprise (uv-only)
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md alembic.ini ./
COPY alembic ./alembic
COPY src ./src

RUN uv sync --frozen --no-dev --extra all --extra redis --no-editable

# --- Final Slim Runtime Stage ---
FROM python:3.11-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home appuser

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app
COPY apps ./apps
COPY alembic.ini ./
COPY alembic ./alembic

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV LOG_FORMAT=json
ENV DOCUMENTS_DIR=/app/data/documents

RUN mkdir -p /app/data/documents /app/storage \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000
EXPOSE 7861

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "rag_xper.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
