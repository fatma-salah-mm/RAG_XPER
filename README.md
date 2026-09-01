# RAG_XPER: Enterprise Multi-Modal Retrieval-Augmented Generation Pipeline

RAG_XPER is an enterprise-grade document question-answering system designed for Arabic and English documents (PDFs, Markdown, plain text, and scanned images). Built on a modular, decoupled architecture, it integrates Qdrant vector search, persistent BM25 lexical retrieval with Reciprocal Rank Fusion (RRF), optical character recognition (OCR), and Chain-of-Thought (CoT) reasoning via large language models.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Capabilities](#key-capabilities)
  - [Modular Chunking Strategies](#modular-chunking-strategies)
  - [Hybrid Retrieval Engine](#hybrid-retrieval-engine)
  - [Multi-Modal Ingestion](#multi-modal-ingestion)
  - [Security and Rate Limiting](#security-and-rate-limiting)
  - [Asynchronous Job Processing](#asynchronous-job-processing)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Execution Modes](#execution-modes)
  - [1. Web Interface (Gradio)](#1-web-interface-gradio)
  - [2. Command-Line Interface (CLI)](#2-command-line-interface-cli)
  - [3. REST API Backend (FastAPI)](#3-rest-api-backend-fastapi)
  - [4. Containerized Deployment (Docker Compose)](#4-containerized-deployment-docker-compose)
- [API Reference](#api-reference)
- [Testing and Quality Assurance](#testing-and-quality-assurance)
- [Configuration Reference](#configuration-reference)

---

## Overview

Traditional RAG pipelines often struggle with scanned multi-page documents, Arabic grammatical variations, legal structure boundaries, and redundant embedding costs. RAG_XPER addresses these challenges by implementing:

- Dual vector store support (Qdrant Server/Embedded and ChromaDB).
- Persistent BM25 index with Arabic normalization, light stemming, and ordinal expansion.
- Content-hash deduplication (SHA-256) to eliminate duplicate vector embedding operations.
- Four distinct chunking strategies tailored to document structure.
- Background asynchronous ingestion jobs with real-time status tracking.
- Enterprise API security with token-based authentication and request validation.

---

## System Architecture

```text
+-----------------------------------------------------------------------------------+
|                                  Client Layer                                     |
|           FastAPI REST API     |     Gradio Web UI     |     Terminal CLI         |
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
| - Modular Chunkers      |  | - Shared RRF Fusion     |  | - Parent-Child Resolver |
| - SHA-256 Hash Dedup    |  | - Payload Filtering     |  | - Source Attribution    |
+-------------------------+  +-------------------------+  +-------------------------+
```

---

## Key Capabilities

### Modular Chunking Strategies

The ingestion engine supports four strategy patterns accessible via configuration or per-request parameters:

1. **Recursive Chunking (`recursive`)**: Standard sliding window with natural paragraph and sentence boundary preservation. Best suited for general literature and reports.
2. **Parent-Child Chunking (`parent_child`)**: Small child chunks (e.g., 300 characters) are indexed for precise semantic search. Upon retrieval, the complete parent chunk (e.g., 1500 characters) is resolved and supplied to the LLM context.
3. **Article-Based Chunking (`article_based`)**: Uses regular expression boundaries to split legal codes, regulations, and contracts strictly along article and clause headings (e.g., `المادة 1`, `Article 1`, `Section 1`).
4. **Auto-Detection (`auto`)**: Analyzes document heuristics across the initial pages to dynamically assign either article-based or recursive chunking.

### Hybrid Retrieval Engine

RAG_XPER employs Reciprocal Rank Fusion (RRF) to combine dense vector rankings from Qdrant with sparse lexical rankings from BM25:

$$\text{RRF Score} = \alpha \cdot \frac{1}{k + \text{Rank}_{\text{Dense}} + 1} + (1 - \alpha) \cdot \frac{1}{k + \text{Rank}_{\text{BM25}} + 1}$$

The BM25 component features Arabic orthographic normalization (unifying forms of Alef, Yaa, Taa Marbuta), diacritic stripping, light prefix/suffix stemming, and number-to-ordinal expansion (e.g., mapping numeric digits to textual Arabic words like `70` to `السبعون` and `سبعين`).

### Multi-Modal Ingestion

- **Digital PDFs**: High-throughput native text layer extraction via PyMuPDF.
- **Scanned Documents**: Automatic fallback to high-resolution rasterization ($2.5\times$ zoom matrix) and EasyOCR/PaddleOCR processing with paragraph line grouping.
- **Markdown & Plain Text**: Direct file ingestion without OCR overhead.
- **Images**: Direct optical extraction for standalone PNG, JPG, and WEBP files.

### Security and Rate Limiting

- Configurable API key authentication via the `X-API-Key` request header.
- Strict MIME-type and extension whitelisting.
- Payload size validation (configurable up to 50 MB).
- Cross-Origin Resource Sharing (CORS) policy enforcement.

### Asynchronous Job Processing

For large files, the API provides non-blocking ingestion endpoints returning a unique `job_id`. Background workers execute the extraction, OCR, chunking, and embedding stages while exposing progress updates ($0\% \rightarrow 100\%$) via status polling endpoints.

### Server-Side Folder Ingestion

Documents staged on the server are indexed without being uploaded through the API. Files placed in `DOCUMENTS_DIR` (bind-mounted into the container at `/app/data/documents`) are indexed by a single request:

```bash
curl -X POST http://localhost:8000/v1/ingest/folder \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"strategy": "auto", "recursive": true}'
```

The endpoint returns `202 Accepted` with a `job_id`. Polling `/v1/jobs/{job_id}` reports progress and, on completion, a per-file breakdown of ingested, skipped, and failed documents. A file that fails to parse is recorded in the report without aborting the remainder of the batch. Requested directories are resolved and rejected if they fall outside `DOCUMENTS_DIR`.

---

## Project Structure

```text
RAG_XPER/
├── src/
│   └── rag_xper/
│       ├── __init__.py
│       ├── config.py                 # Immutable dataclass settings
│       ├── bootstrap.py              # Single decoupled wiring entrypoint
│       ├── core/
│       │   ├── ingestion/
│       │   │   ├── document_extractor.py
│       │   │   ├── ocr_engine.py
│       │   │   └── text_chunker.py
│       │   ├── retrieval/
│       │   │   ├── base_vector_store.py
│       │   │   ├── bm25_retriever.py
│       │   │   ├── hybrid_fusion.py      # Unified RRF implementation
│       │   │   ├── qdrant_store_manager.py
│       │   │   └── vector_store_manager.py
│       │   ├── generation/
│       │   │   ├── llm_interface.py
│       │   │   └── rag_orchestrator.py
│       │   ├── jobs.py                   # Async job manager
│       │   ├── models.py                 # Typed domain models
│       │   └── exceptions.py             # Custom exception hierarchy
│       ├── api/
│       │   ├── __init__.py
│       │   └── app.py                    # Production FastAPI application
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py                   # Interactive CLI application
│       └── utils/
│           ├── __init__.py
│           └── logger.py                 # Structured logger
├── apps/
│   └── gradio_ui/
│       └── app.py                        # Web interface
├── data/
│   └── documents/                        # Server-side ingestion folder (gitignored)
├── tests/
│   ├── test_api.py
│   ├── test_bm25_arabic.py
│   ├── test_chunkers.py
│   ├── test_ingest_folder.py
│   ├── test_orchestrator.py
│   └── test_qdrant_search.py
├── docs/
│   ├── DEPLOYMENT_AWS.md
│   └── PRODUCTION_PLAN.md
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- Git

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/xper-erp/rag.git
   cd rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set your `GEMINI_API_KEY` (or configure local Ollama).

---

## Execution Modes

### 1. Web Interface (Gradio)

Launch the interactive web UI:

```bash
python apps/gradio_ui/app.py
```

Access the interface at `http://localhost:7861`.

### 2. Command-Line Interface (CLI)

Run the CLI tool directly via the registered console script:

```bash
rag-xper
```

Alternatively, invoke subcommands:

```bash
# Ingest a document
rag-xper ingest /path/to/document.pdf --strategy parent_child

# Ingest every supported file in a folder (defaults to DOCUMENTS_DIR)
rag-xper ingest-dir --recursive --strategy auto

# Ask a question
rag-xper ask "What are the contractual obligations under Article 12?"
```

### 3. REST API Backend (FastAPI)

Launch the production REST API server:

```bash
rag-xper-api
```

- Interactive Documentation (Swagger UI): `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/health`
- Metrics Endpoint: `http://localhost:8000/metrics`

### 4. Containerized Deployment (Docker Compose)

Deploy the complete multi-service stack (FastAPI, Redis, and Qdrant Server):

```bash
docker compose -f docker/docker-compose.yml up -d
```

---

## API Reference

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :---: |
| `GET` | `/health` | Liveness check | No |
| `GET` | `/ready` | Readiness check (Qdrant & storage status) | No |
| `GET` | `/version` | Application version | No |
| `GET` | `/metrics` | Operational metrics (queries, ingests, index size) | No |
| `POST` | `/v1/ingest` | Synchronous document upload and indexing | Yes |
| `POST` | `/v1/ingest/async` | Asynchronous upload returning `job_id` (202 Accepted) | Yes |
| `POST` | `/v1/ingest/folder` | Index files staged under `DOCUMENTS_DIR`, returning `job_id` | Yes |
| `GET` | `/v1/jobs/{job_id}` | Check status and progress percentage of an ingestion job | Yes |
| `POST` | `/v1/ask` | Submit a question and retrieve an answer with sources | Yes |
| `GET` | `/v1/documents` | List all indexed files and chunk counts | Yes |
| `DELETE` | `/v1/documents/{filename}` | Delete all indexed chunks associated with a file | Yes |

---

## Testing and Quality Assurance

Run the automated test suite with pytest:

```bash
pytest
```

The test suite covers:
- Unit validation of all chunking strategies (`Recursive`, `ParentChild`, `ArticleBased`, `AutoDetect`).
- Arabic normalization, stemming, and BM25 disk persistence.
- Qdrant storage manager and hybrid RRF retrieval.
- Orchestrator Chain-of-Thought parsing.
- FastAPI endpoints (`/health`, `/version`, `/metrics`, `/v1/documents`, `/v1/jobs`).

---

## Configuration Reference

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `gemini` | Language model provider (`gemini` or `ollama`) |
| `GEMINI_API_KEY` | — | Google AI Studio API key |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Generation model name |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Dense embedding model |
| `EMBEDDING_DIM` | `3072` | Embedding vector dimension (3072 for Gemini, 768 for Ollama) |
| `VECTOR_STORE_TYPE` | `qdrant` | Vector database backend (`qdrant` or `chromadb`) |
| `QDRANT_STORAGE_PATH` | `./storage/qdrant_db` | Storage path for embedded local Qdrant |
| `QDRANT_URL` | — | Remote Qdrant server URL (leave empty for embedded) |
| `COLLECTION_NAME` | `rag_xper_documents` | Target vector collection name |
| `CHUNKING_STRATEGY` | `recursive` | Default chunking strategy |
| `USE_HYBRID_SEARCH` | `true` | Enable hybrid dense + BM25 search |
| `HYBRID_ALPHA` | `0.5` | Weight between dense and lexical search (0.0 to 1.0) |
| `TOP_K` | `6` | Number of final chunks passed to LLM |
| `FETCH_K` | `25` | Number of candidates fetched before fusion |
| `API_KEYS` | — | Comma-separated authorized API keys (leave empty for open access) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum accepted upload size |
| `DOCUMENTS_DIR` | `./data/documents` | Server-side folder scanned by `/v1/ingest/folder` |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## Documentation

- [AWS EC2 Deployment Guide](docs/DEPLOYMENT_AWS.md) — provisioning, Docker Compose, TLS, backups, and operations
- [Production Plan](docs/PRODUCTION_PLAN.md) — phase status, open gaps, and the hardening roadmap

---

## License

This project is licensed under the terms of the MIT License.
