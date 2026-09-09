# RAG_XPER: Enterprise Multi-Modal Retrieval-Augmented Generation Pipeline

RAG_XPER is an enterprise-grade document question-answering system designed for Arabic and English documents (PDFs, Markdown, plain text, and scanned images). Built on a modular, decoupled architecture, it integrates Qdrant vector search, persistent BM25 lexical retrieval with Reciprocal Rank Fusion (RRF), optical character recognition (OCR), an In-Memory Query Cache layer, a MySQL document catalog, and Chain-of-Thought (CoT) reasoning via large language models.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Capabilities](#key-capabilities)
  - [Cross-Page Legal Chunking](#cross-page-legal-chunking)
  - [Hybrid Retrieval Engine](#hybrid-retrieval-engine)
  - [In-Memory Query Cache Layer](#in-memory-query-cache-layer)
  - [MySQL Document and Query Catalog](#mysql-document-and-query-catalog)
  - [Enterprise Security and Document Identity](#enterprise-security-and-document-identity)
  - [Asynchronous Background Ingestion](#asynchronous-background-ingestion)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Execution Modes](#execution-modes)
  - [1. Enterprise Web Dashboard](#1-enterprise-web-dashboard)
  - [2. REST API Backend (FastAPI)](#2-rest-api-backend-fastapi)
  - [3. Developer Web Interface (Gradio)](#3-developer-web-interface-gradio)
  - [4. Command-Line Interface (CLI)](#4-command-line-interface-cli)
  - [5. Containerized Deployment (Docker Compose)](#5-containerized-deployment-docker-compose)
- [API Reference](#api-reference)
- [Testing and Quality Assurance](#testing-and-quality-assurance)
- [Retrieval Quality Benchmark](#retrieval-quality-benchmark)
- [Configuration Reference](#configuration-reference)

---

## Overview

Traditional RAG pipelines often struggle with scanned multi-page documents, Arabic grammatical variations, legal structure boundaries across page margins, and redundant embedding costs. RAG_XPER addresses these challenges by implementing:

- Dual vector store architecture (Qdrant Server/Embedded and ChromaDB).
- Persistent BM25 index with Arabic normalization, light stemming, and ordinal expansion.
- Document identity deduplication (`doc_id`, `file_hash`) preventing redundant vector embedding.
- Cross-page article chunking that extracts complete legal articles spanning multiple pages.
- Sub-5ms response acceleration via a normalized in-memory Query Cache.
- Relational persistence in MySQL (`rag_xper_db`) with SQLite fallback.
- Fail-closed security with constant-time API key verification (`secrets.compare_digest`).
- Background asynchronous ingestion jobs with real-time status tracking.

---

## System Architecture

```text
+-----------------------------------------------------------------------------------+
|                                  Client Layer                                     |
|     XPER Web Dashboard     |     FastAPI REST API     |     Terminal CLI          |
|      (apps/web_dashboard)  |      (src/rag_xper/api)  |     (src/rag_xper/cli)    |
+------------------------------------------+----------------------------------------+
                                           |
+------------------------------------------v----------------------------------------+
|                                Bootstrap & Orchestration                          |
|             (Configuration Validation, Factory Wiring, Pipeline Coordination)     |
+--------------------+---------------------+-------------------+--------------------+
                     |                     |                   |
+--------------------v----+  +-------------v-----------+  +----v--------------------+
|     Ingestion Layer     |  |     Retrieval Layer     |  |    Generation Layer     |
| - PyMuPDF / Text Parser |  | - Qdrant (Rust Vector)  |  | - Gemini / Ollama LLM   |
| - OCR (EasyOCR/Paddle)  |  | - Persisted BM25 Index  |  | - Chain-of-Thought (CoT)|
| - Cross-Page Chunker    |  | - Shared RRF Fusion     |  | - Strict Citation Engine|
| - SHA-256 Hash Dedup    |  | - Metadata Filtering    |  | - In-Memory Query Cache |
+--------------------+----+  +-------------+-----------+  +----+--------------------+
                     |                     |                   |
                     +---------------------+-------------------+
                                           |
+------------------------------------------v----------------------------------------+
|                              Persistence & Storage Layer                          |
|         Qdrant Storage     |     MySQL Database     |     BM25 Disk Pickle        |
|      (storage/qdrant_db)   |    (rag_xper_db)       |    (storage/bm25_index.pkl) |
+-----------------------------------------------------------------------------------+
```

---

## Key Capabilities

### Cross-Page Legal Chunking

The ingestion engine supports modular strategies configured via environment or per-request parameters:

1. **Auto-Detection (`auto`) [Default]**: Inspects document contents across up to 15 pages to dynamically select either article-based or recursive chunking.
2. **Article-Based Chunking (`article_based`)**: Joins multi-page documents with boundary markers and splits text strictly along legal article and clause boundaries (e.g., `المادة 1`, `Article 1`, `Section 1`). Articles spanning across page breaks are kept intact in a single chunk with extracted `article_number` metadata.
3. **Parent-Child Chunking (`parent_child`)**: Small child chunks (e.g., 300 characters) are indexed for precise semantic search. Upon retrieval, the complete parent chunk (e.g., 1500 characters) is resolved and supplied to the LLM context.
4. **Recursive Chunking (`recursive`)**: Standard sliding window with natural paragraph and sentence boundary preservation.

### Hybrid Retrieval Engine

RAG_XPER employs Reciprocal Rank Fusion (RRF) to combine dense vector rankings from Qdrant with sparse lexical rankings from BM25:

$$\text{RRF Score} = \alpha \cdot \frac{1}{k + \text{Rank}_{\text{Dense}} + 1} + (1 - \alpha) \cdot \frac{1}{k + \text{Rank}_{\text{BM25}} + 1}$$

The BM25 component features Arabic orthographic normalization (unifying forms of Alef, Yaa, Taa Marbuta), diacritic stripping, light prefix/suffix stemming, and number-to-ordinal expansion (e.g., mapping numeric digits to textual Arabic words like `70` to `السبعون` and `سبعين`).

### In-Memory Query Cache Layer

A high-performance LRU cache layer intercepts incoming questions before execution:
- Normalizes Arabic variations (Alef variants, Taa Marbuta, diacritics) so spelling variants hit the same cache entry.
- Sub-5ms response time on cache hits without consuming LLM API quotas.
- Automatically invalidated upon new document ingestion or document deletion.

### MySQL Document and Query Catalog

The system provides a relational schema in `rag_xper_db` (UTF-8 Multi-byte `utf8mb4_unicode_ci`):
- **`books` Table**: Catalogs document titles, authors, categories, file paths, chunk counts, and indexing strategies.
- **`query_logs` Table**: Records chat sessions, questions, CoT reasoning steps, answers, retrieved sources, execution latencies, and cache status.
- Resilient fallback: Automatically connects to local SQLite storage if MySQL is not configured.

### Enterprise Security and Document Identity

- Constant-time API key verification using `secrets.compare_digest`.
- Fail-closed initialization: When `REQUIRE_AUTH=true`, the service refuses to start if `API_KEYS` is empty.
- Stable document identity: `doc_id` generated via deterministic `uuid5(filename + file_bytes_hash)`.
- Synchronized deletion: Deleting a document removes its points from Qdrant, BM25, and MySQL simultaneously.
- Docker ports hardened: Qdrant (6333/6334) and Redis (6379) isolated within the internal network.

### Asynchronous Background Ingestion

- Ingestion of large documents runs via an in-process background worker queue.
- Progress tracked in increments (0% to 100%) through polling `/v1/jobs/{job_id}`.
- Directory scanning: Ingest entire folders via `POST /v1/ingest/folder` with batch failure isolation.

---

## Project Structure

```text
RAG_XPER/
├── apps/
│   ├── web_dashboard/           # Enterprise XPER Web Interface (HTML, CSS, JS)
│   └── gradio_ui/               # Developer exploration UI
├── src/
│   └── rag_xper/                # Core production Python package
│       ├── api/
│       │   └── app.py           # FastAPI application and route handlers
│       ├── cli/
│       │   └── main.py          # Terminal CLI implementation
│       ├── core/
│       │   ├── cache.py         # In-Memory Query Cache with Arabic normalization
│       │   ├── db/              # MySQL models, session, and CRUD service
│       │   ├── generation/      # LLM interfaces and RAG orchestrator
│       │   ├── ingestion/       # Extractors, OCR engine, and text chunkers
│       │   └── retrieval/       # BM25 retriever, Qdrant store, and RRF fusion
│       ├── utils/
│       │   └── logger.py        # Structured JSON and standard logging
│       ├── bootstrap.py         # Component factory wiring
│       └── config.py            # Centralized settings and validation
├── tests/                       # 55 automated unit, stress, and security tests
│   └── eval/                    # Retrieval evaluation suite and dataset
├── scripts/
│   └── init_mysql.sql           # MySQL database initialization script
├── docker/
│   ├── Dockerfile               # Multi-stage container definition with OCR
│   └── docker-compose.yml       # Production composition with network isolation
├── pyproject.toml               # Package configuration and dependencies
└── README.md                    # Technical documentation
```

---

## Installation and Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Git
- Qdrant (optional, embedded storage supported natively)
- MySQL 8.0+ (optional, SQLite fallback supported)

### Step 1: Clone and Environment Setup

```bash
git clone https://github.com/xper-erp/rag.git
cd rag
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### Step 2: Install Package with Dependencies

```bash
pip install --upgrade pip
pip install -e ".[all]"
```

### Step 3: Environment Configuration

Copy the sample environment file and configure credentials:

```bash
cp .env.example .env
```

Key environment variables:

```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
EMBEDDING_DIM=3072

VECTOR_STORE_TYPE=qdrant
QDRANT_STORAGE_PATH=./storage/qdrant_db
COLLECTION_NAME=rag_xper_documents

CHUNKING_STRATEGY=auto
USE_HYBRID_SEARCH=true
HYBRID_ALPHA=0.5
TOP_K=6

REQUIRE_AUTH=false
API_KEYS=secret_key_1,secret_key_2
CACHE_ENABLED=true
```

---

## Execution Modes

### 1. Enterprise Web Dashboard

Launch the FastAPI application:

```bash
rag-xper-api
```

Open a web browser and navigate to:
- **Web Dashboard:** `http://localhost:8000/ui`
- **API Documentation (Swagger):** `http://localhost:8000/docs`

The Web Dashboard features:
- Arabic Right-To-Left (RTL) interface with English language toggle.
- Conversational interface with collapsible Chain-of-Thought reasoning.
- Citation cards displaying source filenames and extracted page numbers.
- Sub-5ms cache indicator for cached query responses.
- Modal document upload with real-time background indexing progress.
- MySQL Document Catalog table with search and deletion controls.

### 2. REST API Backend (FastAPI)

Run the server directly via Python:

```bash
python api.py
```

### 3. Developer Web Interface (Gradio)

For testing and parameter experimentation:

```bash
rag-xper-ui
```

Access at `http://localhost:7861`.

### 4. Command-Line Interface (CLI)

```bash
# Ingest a document
rag-xper ingest /path/to/document.pdf --strategy auto

# Ingest an entire directory
rag-xper ingest-dir ./data/documents --recursive

# Ask a question
rag-xper ask "ما هي شروط قبول شهادة الشهود؟" --top-k 6
```

### 5. Containerized Deployment (Docker Compose)

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

---

## API Reference

### Key Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/v1/ask` | Query knowledge base with CoT reasoning and cache acceleration |
| `POST` | `/v1/ingest` | Synchronous ingestion for documents (PDF, TXT, MD, Images) |
| `POST` | `/v1/ingest/async` | Asynchronous ingestion returning a job ID |
| `POST` | `/v1/ingest/folder` | Server-side directory ingestion |
| `GET` | `/v1/jobs/{job_id}` | Poll background ingestion progress (0-100%) |
| `GET` | `/v1/books` | List cataloged books and documents from MySQL |
| `GET` | `/v1/documents` | List unique indexed documents across vector store and BM25 |
| `DELETE`| `/v1/documents/{filename}` | Delete all points for a document across Qdrant, BM25, and MySQL |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe verifying vector store dimension match |
| `GET` | `/metrics` | Operational metrics (uptime, queries, cache hits, chunk counts) |

### Query Example

```bash
curl -X POST "http://localhost:8000/v1/ask" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "question": "ما هي شروط قبول شهادة الشهود وفق نظام الإثبات؟",
    "top_k": 6
  }'
```

### Query Response Example

```json
{
  "answer": "وفقاً للمادة السبعين من نظام الإثبات، تشترط أهلية الشاهد وأن يكون متمتعاً بالأهلية المعتبرة نظاماً وقت أداء الشهادة [1].",
  "reasoning": "1. تم فحص نص المادة 70 في وثيقة nizam_alithbat.pdf (صفحة 18).\n2. النص يحدد شروط أهلية الشاهد وعدم جواز رد الشهادة إلا لأسباب نظامية محددة.",
  "sources": [
    {
      "source": "nizam_alithbat.pdf",
      "page": 18,
      "strategy": "article_based",
      "score": 0.0325,
      "text": "المادة السبعون: يجب أن يكون الشاهد أهلاً لأداء الشهادة..."
    }
  ],
  "query": "ما هي شروط قبول شهادة الشهود وفق نظام الإثبات؟",
  "is_cached": false,
  "execution_time_ms": 142.8
}
```

---

## Retrieval Quality Benchmark

RAG_XPER includes an automated retrieval evaluation benchmark evaluated on real Arabic and English legal corpora:

| Metric | Target | Result | Status |
| :--- | :--- | :--- | :--- |
| **Recall@6** | $\ge 85\%$ | **92.0%** (23/25) | Passed |
| **MRR (Mean Reciprocal Rank)** | $\ge 0.70$ | **0.8800** | Passed |
| **Cache Hit Latency** | $< 10\text{ ms}$ | **3.2 ms** | Passed |
| **Automated Test Suite** | 100% Pass | **55 / 55 Passed** | Passed |

Run the evaluation benchmark locally:

```bash
pytest tests/eval/test_eval_retrieval.py
```

---

## Testing and Quality Assurance

The test suite covers unit logic, edge cases, security controls, and stress scenarios:

```bash
pytest
```

Test coverage categories:
- `tests/test_team_action_plan_waves.py`: Fail-closed auth, cross-page chunking, strict prompt parser, score thresholding, structured JSON logging.
- `tests/eval/test_eval_retrieval.py`: Retrieval Recall@6 and MRR validation.
- `tests/test_mysql_and_cache.py`: Cache hit/TTL and MySQL CRUD operations.
- `tests/test_stress_chunking.py`: Long strings, micro chunk sizes, multilingual Unicode.
- `tests/test_stress_api_security.py`: File extension whitelisting, script rejection, payload size limits.
- `tests/test_stress_retrieval_and_bm25.py`: Arabic diacritics, number expansion, and RRF edge cases.
- `tests/test_stress_orchestrator_edge_cases.py`: Fallback paths, missing files, and parent-child context deduplication.

---

## Configuration Reference

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `LLM_PROVIDER` | string | `gemini` | Language model provider (`gemini` or `ollama`) |
| `GEMINI_API_KEY` | string | `""` | Google Gemini API key |
| `GEMINI_MODEL` | string | `gemini-3.5-flash-lite` | Generation model name |
| `EMBEDDING_DIM` | integer | `3072` | Embedding vector dimension (3072 for Gemini, 768 for Ollama) |
| `VECTOR_STORE_TYPE` | string | `qdrant` | Vector database backend (`qdrant` or `chromadb`) |
| `QDRANT_STORAGE_PATH`| string | `./storage/qdrant_db` | Local directory for embedded Qdrant storage |
| `COLLECTION_NAME` | string | `rag_xper_documents` | Qdrant collection name |
| `CHUNKING_STRATEGY` | string | `auto` | Default chunking strategy (`auto`, `article_based`, `parent_child`, `recursive`) |
| `CHUNK_SIZE` | integer | `1000` | Target characters per chunk for recursive chunker |
| `CHUNK_OVERLAP` | integer | `150` | Overlap characters between consecutive chunks |
| `USE_HYBRID_SEARCH` | boolean | `true` | Combine dense vector and sparse BM25 search |
| `HYBRID_ALPHA` | float | `0.5` | Weight for dense vector search in RRF (0.0 to 1.0) |
| `TOP_K` | integer | `6` | Number of chunks supplied to the LLM context |
| `MIN_RETRIEVAL_SCORE`| float | `0.0` | Minimum score threshold for retrieved passages |
| `REQUIRE_AUTH` | boolean | `false` | When true, server startup fails if API_KEYS is empty |
| `API_KEYS` | string | `""` | Comma-separated list of authorized API keys |
| `MAX_UPLOAD_SIZE_MB`| integer | `50` | Maximum allowable file upload size |
| `CACHE_ENABLED` | boolean | `true` | Enable in-memory LRU Query Cache |
| `CACHE_TTL_SECONDS` | integer | `3600` | Cache time-to-live in seconds |
| `MYSQL_HOST` | string | `None` | MySQL database host (falls back to SQLite if empty) |
| `MYSQL_DATABASE` | string | `rag_xper_db` | MySQL database name |
| `LOG_FORMAT` | string | `text` | Logging format (`text` or `json`) |
