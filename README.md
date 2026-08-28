# 🚀 RAG_XPER: Enterprise Multi-Modal RAG Pipeline

منظومة متقدمة ومستقلة للذكاء الاصطناعي واسترجاع البيانات (RAG) تدعم:
- ⚡ **قاعدة بيانات المتجهات Qdrant:** محرك فائق السرعة مبني بلغة Rust ومخزن محلياً على جهازك مجاناً وبدون سيرفرات خارجية.
- 🧩 **استراتيجيات تقطيع اختيارية ومتعددة (Modular Chunking):**
  1. `Recursive`: عادي للكتب والمستندات العامة (الافتراضي).
  2. `Parent-Child`: تقطيع هرمي (Small-to-Big) يبحث في القطع الصغيرة ويرسل النص الأب الكامل للنموذج.
  3. `Article-Based`: تقطيع مخصص للأنظمة والقوانين واللوائح والعقود (مادة بمادة).
  4. `Auto-Detect`: فحص ذكي وتلقائي لنوع المستند لاختيار أفضل استراتيجية تقطيع.
- 🔍 **بحث هجين دقيق (Hybrid Search):** دمج BM25 باللغة العربية مع المتجهات الكثيفة عبر خوارزمية Reciprocal Rank Fusion (RRF).
- 👁️ **محرك OCR مجمع السطور:** استخراج فائق الدقة للصور والمستندات الممسوحة ضوئياً عبر EasyOCR مع `paragraph=True` و `render_zoom=2.5`.
- 🧠 **تفكير متسلسل (Chain-of-Thought):** تحليل منطقي يمنع الهلوسة ويحدد المصادر وأرقام الصفحات بدقة.

---

## 🏃‍♂️ طرق التشغيل:

### 1. واجهة الويب التفاعلية (Gradio Web UI) 🌐
```powershell
python app.py
```
ثم افتحي المتصفح على: `http://localhost:7861`

---

### 2. واجهة الترمينال التفاعلية (CLI) 💻
```powershell
python main.py
```

---

### 3. خادم الـ REST API (FastAPI) 🔌
```powershell
uvicorn api:app --reload --port 8000
```

---

## ⚙️ التبديل بين قواعد البيانات واستراتيجيات التقطيع:

في ملف `.env`:
```ini
# قاعدة البيانات: qdrant أو chromadb
VECTOR_STORE_TYPE=qdrant

# استراتيجية التقطيع: recursive | parent_child | article_based | auto
CHUNKING_STRATEGY=recursive
```
