"""
tests/test_ingest_folder.py for RAG_XPER

Covers server-side folder ingestion: the orchestrator batch loop, the path guard
that keeps requests inside DOCUMENTS_DIR, and the async job contract.
"""
from importlib import import_module

import pytest
from fastapi.testclient import TestClient

from rag_xper.api.app import app
from rag_xper.core.exceptions import DocumentExtractionError
from rag_xper.core.generation.rag_orchestrator import SUPPORTED_EXTENSIONS, RAGOrchestrator
from rag_xper.core.ingestion.document_extractor import DocumentExtractor
from rag_xper.core.ingestion.ocr_engine import OCREngine

# rag_xper.api re-exports the FastAPI instance as `app`, so reach the module explicitly.
api_module = import_module("rag_xper.api.app")


class _StubStore:
    """Minimal vector store double: records upserts, never deduplicates."""

    def __init__(self):
        self.chunks = []

    def upsert_chunks(self, chunks):
        self.chunks.extend(chunks)
        return len(chunks)

    def is_file_ingested(self, file_path, content_hash=None):
        return False

    def similarity_search(self, query, top_k=4):
        return []

    def hybrid_search(self, query, top_k=6, fetch_k=25, alpha=0.5, rrf_k=60):
        return []

    def delete_file(self, file_path):
        return 0


class _StubLLM:
    def generate(self, prompt):
        return "Answer: ok"

    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]


def _orchestrator(settings=None):
    return RAGOrchestrator(
        extractor=DocumentExtractor(),
        ocr=OCREngine(),
        vector_store=_StubStore(),
        llm=_StubLLM(),
        settings=settings,
    )


# --- Orchestrator batch behaviour -------------------------------------------
def test_supported_extensions_cover_all_ingestible_types():
    assert {".pdf", ".md", ".txt", ".png"} <= SUPPORTED_EXTENSIONS


def test_ingest_directory_rejects_missing_folder():
    with pytest.raises(DocumentExtractionError):
        _orchestrator().ingest_directory("./definitely-not-a-real-folder")


def test_ingest_directory_indexes_text_files(tmp_path):
    (tmp_path / "one.txt").write_text("المادة الأولى تنص على أحكام عامة.", encoding="utf-8")
    (tmp_path / "two.md").write_text("# Article Two\n\nSecond provision.", encoding="utf-8")

    report = _orchestrator().ingest_directory(str(tmp_path))

    assert report["ingested"] == 2
    assert report["failed"] == 0
    assert report["total_chunks"] > 0
    assert {f["file"] for f in report["files"]} == {"one.txt", "two.md"}


def test_ingest_directory_skips_unsupported_extensions(tmp_path):
    (tmp_path / "keep.txt").write_text("محتوى مدعوم", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("a,b,c", encoding="utf-8")
    (tmp_path / "ignore.docx").write_bytes(b"binary")

    report = _orchestrator().ingest_directory(str(tmp_path))

    assert [f["file"] for f in report["files"]] == ["keep.txt"]


def test_ingest_directory_is_not_recursive_by_default(tmp_path):
    (tmp_path / "top.txt").write_text("المستوى الأول", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.txt").write_text("المستوى الثاني", encoding="utf-8")

    shallow = _orchestrator().ingest_directory(str(tmp_path))
    assert [f["file"] for f in shallow["files"]] == ["top.txt"]

    deep = _orchestrator().ingest_directory(str(tmp_path), recursive=True)
    assert {f["file"] for f in deep["files"]} == {"top.txt", "deep.txt"}


def test_ingest_directory_records_failures_without_aborting(tmp_path):
    # A zero-byte PDF cannot be parsed, but it must not stop the healthy files.
    (tmp_path / "broken.pdf").write_bytes(b"")
    (tmp_path / "good.txt").write_text("نص سليم", encoding="utf-8")

    report = _orchestrator().ingest_directory(str(tmp_path))

    statuses = {f["file"]: f["status"] for f in report["files"]}
    assert statuses["good.txt"] == "ingested"
    assert statuses["broken.pdf"] in {"failed", "skipped"}
    assert report["ingested"] == 1


def test_ingest_directory_reports_progress(tmp_path):
    for i in range(3):
        (tmp_path / f"doc{i}.txt").write_text(f"مستند رقم {i}", encoding="utf-8")

    seen = []
    _orchestrator().ingest_directory(str(tmp_path), progress_callback=lambda d, t: seen.append((d, t)))

    assert seen == [(1, 3), (2, 3), (3, 3)]


# --- API contract ------------------------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    from dataclasses import replace

    # Settings is a frozen dataclass, so swap the whole object rather than a field.
    monkeypatch.setattr(
        api_module, "settings", replace(api_module.settings, documents_dir=str(tmp_path))
    )
    return TestClient(app)


def test_folder_ingest_rejects_paths_outside_documents_dir(client):
    response = client.post("/v1/ingest/folder", json={"directory": "/etc"})
    assert response.status_code == 403


def test_folder_ingest_returns_404_for_missing_subfolder(client, tmp_path):
    response = client.post("/v1/ingest/folder", json={"directory": str(tmp_path / "nope")})
    assert response.status_code == 404


def test_folder_ingest_accepts_job_and_reports_result(client, tmp_path, monkeypatch):
    (tmp_path / "law.txt").write_text("المادة الخامسة", encoding="utf-8")
    monkeypatch.setattr(api_module, "get_orchestrator", _orchestrator)

    accepted = client.post("/v1/ingest/folder", json={})
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]

    # TestClient runs background tasks before returning, so the job is already done.
    status = client.get(f"/v1/jobs/{job_id}")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "completed"
    assert body["progress"] == 100
    assert body["details"]["ingested"] == 1
