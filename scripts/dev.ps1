# RAG_XPER developer helper (Windows PowerShell) — uses uv only

param(

    [Parameter(Position = 0)]

    [ValidateSet("install", "sync", "lock", "test", "test-cov", "lint", "format", "security", "audit", "migrate", "docker-build", "all", "run-api", "run-ui", "setup")]

    [string]$Command = "help"

)



$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Set-Location $Root



function Show-Help {

    Write-Host @"

RAG_XPER developer commands (uv):

  .\scripts\dev.ps1 install      - uv sync --extra all --frozen

  .\scripts\dev.ps1 sync         - same as install

  .\scripts\dev.ps1 lock         - regenerate uv.lock

  .\scripts\dev.ps1 setup        - sync + pre-commit hooks

  .\scripts\dev.ps1 test         - uv run pytest

  .\scripts\dev.ps1 test-cov     - pytest with coverage gate

  .\scripts\dev.ps1 lint         - uv run ruff check

  .\scripts\dev.ps1 format       - uv run ruff fix + format

  .\scripts\dev.ps1 security     - bandit scan

  .\scripts\dev.ps1 audit        - pip-audit

  .\scripts\dev.ps1 migrate      - alembic upgrade head

  .\scripts\dev.ps1 docker-build - build production Docker image

  .\scripts\dev.ps1 all          - sync + lint + test-cov + security + audit

  .\scripts\dev.ps1 run-api      - uv run rag-xper-api

  .\scripts\dev.ps1 run-ui       - uv run rag-xper-ui

"@

}



switch ($Command) {

    "install" { uv sync --extra all --frozen }

    "sync" { uv sync --extra all --frozen }

    "lock" { uv lock }

    "setup" {

        uv sync --extra all --frozen

        uv run pre-commit install

    }

    "test" { uv run pytest }

    "test-cov" { uv run pytest --cov=rag_xper --cov-report=term-missing --cov-fail-under=65 }

    "lint" { uv run ruff check src tests }

    "format" {

        uv run ruff check --fix src tests

        uv run ruff format src tests

    }

    "security" { uv run bandit -r src/rag_xper -ll }

    "audit" { uv run pip-audit --ignore-vuln PYSEC-2026-311 --ignore-vuln PYSEC-2026-3813 --ignore-vuln PYSEC-2026-3814 --ignore-vuln PYSEC-2026-3815 }

    "migrate" { uv run alembic upgrade head }

    "docker-build" { docker build -f docker/Dockerfile -t rag-xper:latest . }

    "all" {

        uv sync --extra all --frozen

        uv run ruff check src tests

        uv run pytest --cov=rag_xper --cov-report=term-missing --cov-fail-under=65

        uv run bandit -r src/rag_xper -ll

        uv run pip-audit --ignore-vuln PYSEC-2026-311 --ignore-vuln PYSEC-2026-3813 --ignore-vuln PYSEC-2026-3814 --ignore-vuln PYSEC-2026-3815

    }

    "run-api" { uv run rag-xper-api }

    "run-ui" { uv run rag-xper-ui }

    default { Show-Help }

}

