"""Tests for production operations: Prometheus metrics, jobs, migrations helpers."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag_xper.api.app import app
from rag_xper.api.metrics import collect_metrics, render_prometheus
from rag_xper.core.jobs import IngestionJob, JobStatus, MemoryJobManager, _job_from_dict, _job_to_dict

client = TestClient(app)


def test_prometheus_metrics_format():
    metrics = collect_metrics()
    body = render_prometheus(metrics)
    assert "rag_xper_uptime_seconds" in body
    assert "rag_xper_total_queries" in body
    assert "# TYPE rag_xper_cache_hits gauge" in body
    assert 'rag_xper_vector_store_type{type="' in body


def test_prometheus_metrics_endpoint():
    response = client.get("/metrics/prometheus")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "rag_xper_uptime_seconds" in response.text


def test_job_serialization_roundtrip():
    job = IngestionJob(
        job_id="job-1",
        filename="doc.pdf",
        status=JobStatus.PROCESSING,
        progress=50,
        chunks_ingested=3,
        strategy_used="auto",
        details={"files": ["a.pdf"]},
    )
    restored = _job_from_dict(_job_to_dict(job))
    assert restored.job_id == job.job_id
    assert restored.status == JobStatus.PROCESSING
    assert restored.progress == 50
    assert restored.details == {"files": ["a.pdf"]}


def test_memory_job_manager_lifecycle():
    manager = MemoryJobManager()
    job = manager.create_job("report.pdf", strategy="article_based")
    assert job.status == JobStatus.PENDING

    manager.update_progress(job.job_id, 40, JobStatus.PROCESSING)
    updated = manager.get_job(job.job_id)
    assert updated is not None
    assert updated.progress == 40
    assert updated.status == JobStatus.PROCESSING

    manager.complete_job(job.job_id, chunks_ingested=12, details={"ok": True})
    done = manager.get_job(job.job_id)
    assert done is not None
    assert done.status == JobStatus.COMPLETED
    assert done.chunks_ingested == 12
