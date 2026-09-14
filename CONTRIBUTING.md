# Contributing to RAG_XPER

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- Git

## Setup

```bash
git clone <repo-url>
cd RAG_XPER
uv sync --extra all
cp .env.example .env
```

On Windows you can also use:

```powershell
.\scripts\dev.ps1 setup
```

## Official entry points

All commands use `uv run` (or the Makefile / `dev.ps1` wrappers):

| Task | Command |
|------|---------|
| REST API + Web Dashboard | `uv run rag-xper-api` |
| Gradio developer UI | `uv run rag-xper-ui` |
| CLI ingest / ask | `uv run rag-xper ingest ...` / `uv run rag-xper ask "..."` |
| Interactive demo CLI | `uv run rag-xper interactive` |

Root scripts (`api.py`, `app.py`, `main.py`) are deprecated compatibility shims.

## Development workflow

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check src tests

# Format
uv run ruff format src tests

# Pre-commit (optional)
uv run pre-commit install
uv run pre-commit run --all-files

# After changing dependencies in pyproject.toml
uv lock
uv sync --extra all --frozen
```

## Pull requests

1. Branch from `main`
2. Keep changes focused
3. Commit `uv.lock` when dependencies change
4. Run `uv run pytest` and `uv run ruff check src tests` before opening a PR
5. Update README or `docs/` when behavior or setup changes

## Project layout

- `src/rag_xper/` — production Python package
- `src/rag_xper/api/routes/` — FastAPI route modules
- `apps/web_dashboard/` — static enterprise UI served at `/ui`
- `tests/` — unit, stress, security, and eval tests
- `uv.lock` — locked dependency versions (committed)
- `docs/` — deployment and architecture notes
