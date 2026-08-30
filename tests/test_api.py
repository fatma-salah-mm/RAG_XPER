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
