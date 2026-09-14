# RAG_XPER Architecture

## Overview

RAG_XPER is a modular Retrieval-Augmented Generation pipeline for Arabic and English documents. All runtime surfaces (API, CLI, Gradio) share the same wiring through `bootstrap.build_orchestrator()`.

## Layer diagram

```text
Client Layer
├── Web Dashboard (apps/web_dashboard)  →  FastAPI /ui
├── REST API (src/rag_xper/api)         →  /v1/*
├── Gradio UI (src/rag_xper/ui)         →  :7861
└── CLI (src/rag_xper/cli)              →  rag-xper

Bootstrap (src/rag_xper/bootstrap.py)
├── LLM (Gemini / Ollama)
├── Vector store factory (Qdrant / ChromaDB)
├── Document extractor + OCR
└── RAGOrchestrator

Core pipeline
├── ingestion/   extract → chunk → embed
├── retrieval/   dense + BM25 → RRF fusion
└── generation/  CoT prompt → LLM answer

Persistence
├── Qdrant or ChromaDB (vectors)
├── BM25 pickle (lexical index)
├── MySQL or SQLite (catalog + query logs)
└── In-memory LRU query cache
```

## API structure

```
src/rag_xper/api/
├── app.py           FastAPI factory, middleware, UI mount
├── dependencies.py  API key auth
├── schemas.py       Pydantic models
├── state.py         Orchestrator singleton + metrics
├── helpers.py       Upload and path guards
├── workers.py       Background ingestion jobs
└── routes/
    ├── health.py
    ├── ingest.py
    ├── ask.py
    └── documents.py
```

## Configuration

Settings load from environment variables and `.env` via `pydantic-settings` (`src/rag_xper/config.py`). `settings.validate()` runs at orchestrator startup and fails fast on missing API keys or invalid chunk sizes.

## Job queue

Background ingestion jobs are tracked in-process (`core/jobs.py`). This suits single-replica deployments. For horizontal scaling, move job state to an external queue (planned in `docs/TEAM_ACTION_PLAN.md`).

## Security model

- Optional API key auth via `X-API-Key` header (`REQUIRE_AUTH`, `API_KEYS`)
- Constant-time key comparison (`secrets.compare_digest`)
- Upload size limits and extension whitelist
- Folder ingestion restricted to `DOCUMENTS_DIR`

## Dependency management

The project uses [uv](https://docs.astral.sh/uv/) exclusively. Locked versions live in `uv.lock`.

```bash
uv sync --extra all          # install runtime + dev deps
uv run pytest                # run tests in the project venv
uv lock                      # refresh lockfile after pyproject changes
```

## Testing strategy

| Suite | Purpose |
|-------|---------|
| `tests/test_*.py` | Unit and integration tests |
| `tests/test_stress_*` | Edge cases and security |
| `tests/eval/` | Retrieval quality benchmark (Recall@6, MRR) |
