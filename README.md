# 🚀 RAG_XPER: Enterprise Multi-Modal RAG Pipeline

منظومة متقدمة ومستقلة للذكاء الاصطناعي واسترجاع البيانات (RAG) تدعم:
- ⚡ **قاعدة بيانات المتجهات Qdrant:** محرك فائق السرعة مبني بلغة Rust (يعمل محلياً Embedded أو عبر السيرفر Remote).
- 🧩 **استراتيجيات تقطيع اختيارية ومتعددة (Modular Chunking):**
  1. `Recursive`: عادي للكتب والمستندات والتقارير (الافتراضي).
  2. `Parent-Child`: تقطيع هرمي (Small-to-Big) يبحث في القطع الصغيرة ويرسل النص الأب الكامل للنموذج.
  3. `Article-Based`: تقطيع مخصص للأنظمة والقوانين واللوائح والعقود (مادة بمادة).
  4. `Auto-Detect`: فحص ذكي وتلقائي لنوع المستند لاختيار أفضل استراتيجية تقطيع.
- 📄 **دعم ملفات متعدد:** PDF (رقمي أو ممسوح ضوئياً)، Markdown (`.md`)، نصوص (`.txt`)، وجميع أنواع الصور (`PNG, JPG, WEBP`).
- 🔍 **بحث هجين فائق الدقة (Hybrid Search):** دمج BM25 باللغة العربية مع المتجهات الكثيفة عبر خوارزمية Reciprocal Rank Fusion (RRF) مع تخزين الفهرس على القرص.
- 🔒 **حماية وأمان الـ API:** دعم مفاتيح المصادقة `API_KEYS`، حظر الملفات الضارة، وفحص الأحجام.
- 🧠 **تفكير متسلسل (Chain-of-Thought):** تحليل منطقي يمنع الهلوسة ويحدد المصادر وأرقام الصفحات بدقة.

---

## 📦 التثبيت (Installation)

```powershell
# التثبيت كحزمة محلية قابلة للتعديل
pip install -e .
```

---

## 🏃‍♂️ طرق التشغيل (Usage Modes)

### 1. واجهة الويب التفاعلية (Gradio Web UI) 🌐
```powershell
python apps/gradio_ui/app.py
```
*(أو مباشرة: `python app.py`)* $\rightarrow$ افتحي المتصفح على: `http://localhost:7861`

---

### 2. واجهة الترمينال (CLI) 💻
```powershell
rag-xper
```
*(أو مباشرة: `python main.py`)*

---

### 3. خادم الـ REST API (FastAPI) 🔌
```powershell
rag-xper-api
```
*(أو: `uvicorn rag_xper.api.app:app --reload --port 8000`)*
- فحص الصحة: `GET http://localhost:8000/health`
- توثيق الـ API التفاعلي (Swagger UI): `http://localhost:8000/docs`

---

### 4. التشغيل عبر Docker Compose 🐳
```powershell
docker compose -f docker/docker-compose.yml up -d
```

---

## 🧪 تشغيل الاختبارات الآلية (Testing)
```powershell
pytest
```
