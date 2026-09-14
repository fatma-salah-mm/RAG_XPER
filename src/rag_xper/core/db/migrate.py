"""Programmatic Alembic migration runner."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)


def run_migrations() -> None:
    """Apply all pending Alembic migrations to the configured database."""
    root = Path(__file__).resolve().parents[4]
    alembic_ini = root / "alembic.ini"
    if not alembic_ini.exists():
        raise FileNotFoundError(f"Alembic config not found: {alembic_ini}")

    cfg = Config(str(alembic_ini))
    logger.info("Running database migrations (alembic upgrade head)...")
    command.upgrade(cfg, "head")
    logger.info("Database migrations complete.")
