.PHONY: all install sync dev test test-cov lint format security audit migrate docker-build run-api run-ui clean lock



install sync:

	uv sync --extra all --frozen



dev: sync

	uv run pre-commit install



lock:

	uv lock



test:

	uv run pytest



test-cov:

	uv run pytest --cov=rag_xper --cov-report=term-missing



lint:

	uv run ruff check src tests



format:

	uv run ruff check --fix src tests

	uv run ruff format src tests



security:

	uv run bandit -r src/rag_xper -ll



audit:
	uv run pip-audit --ignore-vuln PYSEC-2026-311 --ignore-vuln PYSEC-2026-3813 --ignore-vuln PYSEC-2026-3814 --ignore-vuln PYSEC-2026-3815



migrate:

	uv run alembic upgrade head



docker-build:

	docker build -f docker/Dockerfile -t rag-xper:latest .



all: sync lint test-cov security audit



run-api:

	uv run rag-xper-api



run-ui:

	uv run rag-xper-ui



clean:

	rm -rf .pytest_cache .ruff_cache .coverage htmlcov storage/qdrant_db storage/chroma_db

	find . -type d -name __pycache__ -exec rm -rf {} +

