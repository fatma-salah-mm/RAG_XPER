"""Shared helper functions for API route handlers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile

from rag_xper import config
from rag_xper.core.generation.rag_orchestrator import SUPPORTED_EXTENSIONS


def check_supported_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    return suffix


def save_upload_to_temp(file: UploadFile, suffix: str) -> str:
    """Stream an upload to disk, aborting once it exceeds MAX_UPLOAD_SIZE_MB."""
    max_bytes = config.settings.max_upload_size_mb * 1024 * 1024
    written = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                tmp.close()
                Path(tmp_path).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds MAX_UPLOAD_SIZE_MB ({config.settings.max_upload_size_mb} MB).",
                )
            tmp.write(chunk)

    return tmp_path


def resolve_documents_path(directory: str | None) -> Path:
    """Resolve a requested folder, refusing anything outside DOCUMENTS_DIR."""
    base = Path(config.settings.documents_dir).resolve()
    target = base if not directory else Path(directory).resolve()

    if target != base and base not in target.parents:
        raise HTTPException(
            status_code=403,
            detail=f"Directory must be inside DOCUMENTS_DIR ('{base}').",
        )
    if not target.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {target}")
    return target
