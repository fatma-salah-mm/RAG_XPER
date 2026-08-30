# Multi-stage Dockerfile for RAG_XPER Enterprise
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src

RUN pip install --no-cache-dir --user -e .

# --- Final Slim Runtime Stage ---
FROM python:3.11-slim as runner

WORKDIR /app

# Install runtime dependencies for PyMuPDF & OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV LOG_FORMAT=json

EXPOSE 8000
EXPOSE 7861

CMD ["uvicorn", "rag_xper.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
