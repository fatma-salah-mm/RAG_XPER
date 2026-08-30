"""
tests/test_api.py for RAG_XPER
"""
import pytest
from fastapi.testclient import TestClient
from rag_xper.api.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "RAG_XPER"


def test_version():
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "uptime_seconds" in data
    assert "total_queries" in data
    assert "total_ingests" in data


def test_documents_list():
    response = client.get("/v1/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_job_not_found():
    response = client.get("/v1/jobs/non-existent-id")
    assert response.status_code == 404
