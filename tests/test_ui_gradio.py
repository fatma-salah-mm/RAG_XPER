"""
tests/test_ui_gradio.py

Comprehensive tests for the Gradio Web Interface (apps/gradio_ui/app.py).
Covers:
- UI layout and component initialization
- File upload handling with all strategies and file object types
- Already-indexed file detection
- Upload exception resilience
- Query handling with empty, whitespace, and extreme inputs
- History format compatibility (tuples, dicts, None)
- Source rendering under complete, empty, and malformed metadata
- CoT reasoning rendering
- Query exception resilience
- Concurrent multi-user UI interactions
"""
from __future__ import annotations

import concurrent.futures
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

import gradio as gr
from apps.gradio_ui.app import create_ui, handle_file_upload, handle_query
from rag_xper.core.models import Chunk, RAGResponse, RetrievedChunk


# --- 1. UI Creation and Component Architecture ---

def test_ui_creation_and_components():
    """Verify create_ui returns a valid gr.Blocks with all necessary interactive elements."""
    demo = create_ui()
    assert isinstance(demo, gr.Blocks)
    assert demo.title == "RAG_XPER - Enterprise Multi-Modal RAG"

    blocks_types = [type(b).__name__ for b in demo.blocks.values()]
    assert "File" in blocks_types
    assert "Dropdown" in blocks_types
    assert "Button" in blocks_types
    assert "Chatbot" in blocks_types
    assert "Textbox" in blocks_types
    assert "Accordion" in blocks_types
    assert "Markdown" in blocks_types


# --- 2. File Upload Tests & Edge Cases ---

def test_handle_file_upload_none():
    """Uploading None should prompt user immediately with Arabic warning."""
    result = handle_file_upload(None, "🟢 عادي للكتب والمستندات (Recursive)")
    assert "⚠️ يرجى اختيار ملف أولاً" in result


def test_handle_file_upload_strategy_mapping_and_success(tmp_path):
    """Verify each dropdown strategy maps to the correct backend strategy identifier."""
    dummy_file = tmp_path / "test_law.pdf"
    dummy_file.write_text("Dummy legal content", encoding="utf-8")

    strategies = [
        ("🟢 عادي للكتب والمستندات (Recursive)", "recursive"),
        ("🔵 هرمي دقيق جداً (Parent-Child)", "parent_child"),
        ("🟣 مخصص للقوانين واللوائح والعقود (Article-Based)", "article_based"),
        ("🤖 فحص ذكي تلقائي (Auto-Detect)", "auto"),
        ("خيار غير معروف إطلاقاً", "recursive"),  # Fallback to recursive
    ]

    mock_orch = MagicMock()
    mock_orch.ingest_file.return_value = 14

    with patch("apps.gradio_ui.app.get_orchestrator", return_value=mock_orch):
        for label, expected_key in strategies:
            file_obj = SimpleNamespace(name=str(dummy_file))
            res = handle_file_upload(file_obj, label)
            assert "✅ تم استيعاب وفهرسة الملف بنجاح" in res
            assert "test_law.pdf" in res
            assert f"استراتيجية التقطيع: {expected_key}" in res
            assert "14 قطعة" in res

            call_args = mock_orch.ingest_file.call_args
            assert call_args[1]["strategy"] == expected_key


def test_handle_file_upload_already_indexed(tmp_path):
    """When ingest_file returns 0 chunks, UI informs user file was already indexed."""
    dummy_file = tmp_path / "cached_manual.pdf"
    dummy_file.write_text("Old content", encoding="utf-8")

    mock_orch = MagicMock()
    mock_orch.ingest_file.return_value = 0

    with patch("apps.gradio_ui.app.get_orchestrator", return_value=mock_orch):
        res = handle_file_upload(str(dummy_file), "🟢 عادي للكتب والمستندات (Recursive)")
        assert "مفهرس مسبقاً" in res
        assert "cached_manual.pdf" in res


def test_handle_file_upload_exception_resilience():
    """Verify backend ingestion crashes are caught and displayed safely."""
    mock_orch = MagicMock()
    mock_orch.ingest_file.side_effect = RuntimeError("Disk full: write failed")

    with patch("apps.gradio_ui.app.get_orchestrator", return_value=mock_orch):
        file_obj = SimpleNamespace(name="corrupted.pdf")
        res = handle_file_upload(file_obj, "🤖 فحص ذكي تلقائي (Auto-Detect)")
        assert "❌ فشلت معالجة الملف" in res
        assert "Disk full" in res


# --- 3. Chat & Query Tests & Edge Cases ---

def test_handle_query_empty_or_whitespace():
    """Empty or whitespace queries must return immediately without invoking orchestrator."""
    mock_orch = MagicMock()
    with patch("apps.gradio_ui.app.get_orchestrator", return_value=mock_orch):
        q_out, hist_out, reason_out, src_out = handle_query("", [])
        assert q_out == ""
        assert hist_out == []
        assert reason_out == ""
        assert src_out == ""

        q_out, hist_out, reason_out, src_out = handle_query("   \n\t  ", [{"role": "user", "content": "hi"}])
        assert len(hist_out) == 1
        assert mock_orch.query.called is False


def test_handle_query_modern_dict_history():
    """Verify conversation history in dict format (modern Gradio) appends role/content dicts."""
    mock_orch = MagicMock()
    mock_response = RAGResponse(
        answer="الحد الأدنى للأجور هو 6000 جنيه.",
        reasoning="تم استخراج القيمة من نص القرار رقم 5 لسنة 2024.",
        sources=[],
        query="ما هو الحد الأدنى؟",
    )
    mock_orch.query.return_value = mock_response

    with patch("apps.gradio_ui.app.get_orchestrator", return_value=mock_orch):
        init_history = [{"role": "user", "content": "مرحبا"}, {"role": "assistant", "content": "أهلاً بك"}]
        q_out, hist_out, reason_out, src_out = handle_query("ما هو الحد الأدنى؟", init_history)

        assert q_out == ""
        assert len(hist_out) == 4
        assert hist_out[2] == {"role": "user", "content": "ما هو الحد الأدنى؟"}
        assert hist_out[3] == {"role": "assistant", "content": "الحد الأدنى للأجور هو 6000 جنيه."}
        assert reason_out == "تم استخراج القيمة من نص القرار رقم 5 لسنة 2024."
        assert "لا توجد مصادر مطابقة" in src_out


def test_handle_query_legacy_tuple_history():
    """Verify conversation history in legacy tuple format appends (user, bot) tuples cleanly."""
    mock_orch = MagicMock()
    mock_response = RAGResponse(
        answer="إجابة تجريبية",
        reasoning="",
        sources=[],
        query="سؤال جديد",
    )
    mock_orch.query.return_value = mock_response

    with patch("apps.gradio_ui.app.get_orchestrator", return_value=mock_orch):
        init_history = [("سؤال سابق", "إجابة سابقة")]
        q_out, hist_out, reason_out, src_out = handle_query("سؤال جديد", init_history)

        assert len(hist_out) == 2
        assert hist_out[1] == ("سؤال جديد", "إجابة تجريبية")
        assert reason_out == "تم التوليد المباشر بناءً على السياق المسترجع."


def test_handle_query_none_history():
    """Verify passing None as history initializes an empty list and appends without error."""
    mock_orch = MagicMock()
    mock_response = RAGResponse(answer="إجابة", reasoning="", sources=[], query="سؤال")
    mock_orch.query.return_value = mock_response

    with patch("apps.gradio_ui.app.get_orchestrator", return_value=mock_orch):
        q_out, hist_out, reason_out, src_out = handle_query("سؤال", None)
        assert isinstance(hist_out, list)
        assert len(hist_out) == 2
        assert hist_out[0] == {"role": "user", "content": "سؤال"}


def test_handle_query_sources_rendering_and_malformed_metadata():
    """Verify source markdown formatting handles standard, missing, and malformed metadata without crashing."""
    mock_orch = MagicMock()

    c1 = Chunk(
        chunk_id="1",
        text="نص قانوني",
        metadata={"source": "laws/labor_law.pdf", "page": 42, "source_type": "pdf", "strategy": "article_based"},
    )
    c2 = Chunk(
        chunk_id="2",
        text="نص آخر",
        metadata={"source": None, "strategy": None},
    )

    mock_response = RAGResponse(
        answer="إجابة مبنية على المراجع",
        reasoning="تحليل متكامل للمادة 42",
        sources=[
            RetrievedChunk(chunk=c1, score=0.942),
            RetrievedChunk(chunk=c2, score=None),
        ],
        query="ما هي نصوص القانون؟",
    )
    mock_orch.query.return_value = mock_response

    with patch("apps.gradio_ui.app.get_orchestrator", return_value=mock_orch):
        q_out, hist_out, reason_out, src_out = handle_query("ما هي نصوص القانون؟", [])

        assert "labor_law.pdf" in src_out
        assert "صفحة 42" in src_out
        assert "0.942" in src_out
        assert "doc" in src_out


def test_handle_query_orchestrator_exception():
    """Verify unexpected failure in backend query (e.g., LLM rate limit or DB timeout) is captured in UI."""
    mock_orch = MagicMock()
    mock_orch.query.side_effect = TimeoutError("Gemini API timed out after 30s")

    with patch("apps.gradio_ui.app.get_orchestrator", return_value=mock_orch):
        q_out, hist_out, reason_out, src_out = handle_query("سؤال صعب جداً", [])

        assert "❌ حدث خطأ أثناء الاستعلام" in reason_out
        assert "Gemini API timed out" in reason_out
        assert "❌ حدث خطأ أثناء الاستعلام" in hist_out[-1]["content"]


# --- 4. Concurrency Stress Test on UI Handlers ---

def test_ui_handlers_concurrency_stress():
    """Simulate 25 concurrent users querying and uploading files to UI handlers simultaneously."""
    mock_orch = MagicMock()
    mock_orch.query.return_value = RAGResponse(answer="إجابة فورية سريعة", reasoning="تحليل", sources=[], query="سؤال")
    mock_orch.ingest_file.return_value = 5

    with patch("apps.gradio_ui.app.get_orchestrator", return_value=mock_orch):
        def worker(idx: int):
            if idx % 2 == 0:
                q_out, hist_out, reason_out, src_out = handle_query(f"سؤال رقم {idx}", [])
                assert hist_out[1]["content"] == "إجابة فورية سريعة"
            else:
                fake_file = SimpleNamespace(name=f"document_{idx}.pdf")
                res = handle_file_upload(fake_file, "🟢 عادي للكتب والمستندات (Recursive)")
                assert "✅ تم استيعاب وفهرسة الملف بنجاح" in res

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, i) for i in range(25)]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
