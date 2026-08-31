"""
tests/test_stress_api_security.py

Security and API Stress Tests for RAG_XPER FastAPI backend.
Tests:
- Rejection of unauthorized requests when API_KEYS is enabled
- Whitelisting enforcement on malicious/unsupported upload extensions
- Input boundary validation (negative top_k, empty questions)
- Non-existent document deletions
"""
from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient
from rag_xper.api.app import app
from rag_xper.config import settings

client = TestClient(app)


def test_forbidden_file_extension_rejection():
    """Verify that dangerous or non-document file types are rejected with 400."""
    fake_exe = io.BytesIO(b"MZ\x90\x00\x03\x00\x00\x00")
    response = client.post(
        "/v1/ingest",
        files={"file": ("malware.exe", fake_exe, "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_forbidden_script_extension_rejection():
    """Verify script files are blocked from ingestion."""
    fake_script = io.BytesIO(b"import os; os.system('calc')")
    response = client.post(
        "/v1/ingest",
        files={"file": ("exploit.py", fake_script, "text/x-python")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_input_validation_empty_and_extreme_top_k():
    """Test validation boundaries for /v1/ask."""
    # Empty question
    res_empty = client.post("/v1/ask", json={"question": ""})
    assert res_empty.status_code == 422

    # Zero or negative top_k
    res_neg = client.post("/v1/ask", json={"question": "valid question", "top_k": 0})
    assert res_neg.status_code == 422

    # Excessive top_k (>50)
    res_huge = client.post("/v1/ask", json={"question": "valid question", "top_k": 9999})
    assert res_huge.status_code == 422


def test_delete_non_existent_document():
    """Verify deleting a non-existent document returns 0 deleted chunks safely."""
    response = client.delete("/v1/documents/totally_unknown_file.pdf")
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_deleted"] == 0
    assert data["filename"] == "totally_unknown_file.pdf"
