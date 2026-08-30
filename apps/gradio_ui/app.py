"""
apps/gradio_ui/app.py

Gradio Web Interface for RAG_XPER.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import gradio as gr

from rag_xper.bootstrap import build_orchestrator
from rag_xper.config import settings
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)

_orchestrator = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = build_orchestrator()
    return _orchestrator


def handle_file_upload(file_obj, strategy: str) -> str:
    """Ingest uploaded PDF, Markdown, text, or image."""
    if file_obj is None:
        return "⚠️ يرجى اختيار ملف أولاً."

    strat_map = {
        "🟢 عادي للكتب والمستندات (Recursive)": "recursive",
        "🔵 هرمي دقيق جداً (Parent-Child)": "parent_child",
        "🟣 مخصص للقوانين واللوائح والعقود (Article-Based)": "article_based",
        "🤖 فحص ذكي تلقائي (Auto-Detect)": "auto",
    }
    strat_key = strat_map.get(strategy, "recursive")

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    file_name = Path(file_path).name

    try:
        orch = get_orchestrator()
        n_chunks = orch.ingest_file(file_path, strategy=strat_key)
        if n_chunks > 0:
            return (
                f"✅ تم استيعاب وفهرسة الملف بنجاح!\n"
                f"📄 اسم الملف: {file_name}\n"
                f"🧩 استراتيجية التقطيع: {strat_key}\n"
                f"📦 عدد القطع المفهرسة: {n_chunks} قطعة\n"
                f"🗄️ قاعدة البيانات: {settings.vector_store_type.upper()}"
            )
        else:
            return f"ℹ️ الملف '{file_name}' مفهرس مسبقاً في قاعدة بيانات {settings.vector_store_type.upper()} وجاهز للاستعلام فوراً."
    except Exception as exc:
        logger.error("Upload error: %s", exc)
        return f"❌ فشلت معالجة الملف: {exc}"


def handle_query(question: str, history):
    """Execute RAG query and format response with CoT reasoning and sources."""
    if not question or not question.strip():
        return "", history or [], "", ""

    if history is None:
        history = []

    try:
        orch = get_orchestrator()
        response = orch.query(question.strip(), top_k=settings.top_k)

        # Format sources
        sources_md = "### 📚 المصادر المستند إليها:\n"
        if response.sources:
            for i, src in enumerate(response.sources, 1):
                src_name = Path(src.chunk.metadata.get("source", "doc")).name
                page = src.chunk.metadata.get("page", 1)
                stype = src.chunk.metadata.get("source_type", "text")
                strat = src.chunk.metadata.get("strategy", "default")
                sources_md += f"* **[{i}]** `{src_name}` (صفحة {page} | {stype} | {strat}) — **درجة التطابق:** `{src.score:.3f}`\n"
        else:
            sources_md += "*لا توجد مصادر مطابقة.*"

        reasoning_text = response.reasoning or "تم التوليد المباشر بناءً على السياق المسترجع."
        answer_text = response.answer

        if isinstance(history, list) and (len(history) == 0 or isinstance(history[0], dict)):
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer_text})
        else:
            history.append((question, answer_text))

        return "", history, reasoning_text, sources_md

    except Exception as exc:
        logger.error("Query error: %s", exc)
        error_msg = f"❌ حدث خطأ أثناء الاستعلام: {exc}"
        if isinstance(history, list) and (len(history) == 0 or isinstance(history[0], dict)):
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": error_msg})
        else:
            history.append((question, error_msg))
        return "", history, error_msg, ""


def create_ui() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(title="RAG_XPER - Enterprise Multi-Modal RAG") as demo:
        gr.Markdown(
            f"""
            # 🚀 منظومة RAG_XPER (Enterprise Qdrant + Modular RAG)
            مساعد ذكي متعدد اللغات لمعالجة ملفات PDF، Markdown، والنصوص والصور بتقنيات **Qdrant Vector Database** والتقطيع الهرمي والقانوني المتعدد.
            <br>
            **قاعدة المتجهات الحالية:** `{settings.vector_store_type.upper()} ⚡` | **محرك الـ LLM:** `{settings.gemini_model}`
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 رفع وفهرسة المستندات")
                file_input = gr.File(
                    label="اسحبي ملف PDF أو MD أو صورة هنا",
                    file_types=[".pdf", ".md", ".txt", ".markdown", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"],
                )

                strategy_input = gr.Dropdown(
                    label="⚙️ استراتيجية التقطيع (Chunking Strategy)",
                    choices=[
                        "🟢 عادي للكتب والمستندات (Recursive)",
                        "🔵 هرمي دقيق جداً (Parent-Child)",
                        "🟣 مخصص للقوانين واللوائح والعقود (Article-Based)",
                        "🤖 فحص ذكي تلقائي (Auto-Detect)",
                    ],
                    value="🟢 عادي للكتب والمستندات (Recursive)",
                )

                ingest_btn = gr.Button("🚀 ابدأي الـ Ingestion والفهرسة", variant="primary")
                ingest_status = gr.Textbox(label="حالة الاستيعاب", lines=4, interactive=False)

                ingest_btn.click(
                    fn=handle_file_upload,
                    inputs=[file_input, strategy_input],
                    outputs=[ingest_status],
                )

            with gr.Column(scale=2):
                gr.Markdown("### 💬 المحادثة والاستعلام الذكي")
                chatbot = gr.Chatbot(label="سجل الأسئلة والأجوبة", height=450)

                with gr.Row():
                    query_input = gr.Textbox(
                        label="اكتبي سؤالك هنا...",
                        placeholder="ما هي قيمة العقد؟ / ما هي الحالات وفقاً للمادة 70؟",
                        scale=4,
                        lines=1,
                    )
                    ask_btn = gr.Button("إرسال ✉️", variant="primary", scale=1)

                with gr.Accordion("🧠 تحليل الذكاء الاصطناعي (Chain-of-Thought Reasoning)", open=False):
                    reasoning_box = gr.Textbox(label="خطوات التفكير والتحليل", lines=4, interactive=False)

                sources_box = gr.Markdown("### 📚 المصادر ستظهر هنا بعد كل إجابة...")

                # Wire buttons
                ask_btn.click(
                    fn=handle_query,
                    inputs=[query_input, chatbot],
                    outputs=[query_input, chatbot, reasoning_box, sources_box],
                )
                query_input.submit(
                    fn=handle_query,
                    inputs=[query_input, chatbot],
                    outputs=[query_input, chatbot, reasoning_box, sources_box],
                )

        gr.Markdown(
            """
            ---
            💡 **الميزات التقنية:** محرك Qdrant محلي وسحابي • بحث هجين BM25 + Dense • تقطيع هرمي Parent-Child • تعرّف ضوئي EasyOCR • تفكير متسلسل بدون هلوسة.
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
