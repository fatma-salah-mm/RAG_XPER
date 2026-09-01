# RAG_XPER — Production Transformation Plan

> **Document version:** 2.0
> **Status:** Living document — updated as phases land
> **Owner:** Engineering team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Product Vision & Scope](#2-product-vision--scope)
3. [Current State](#3-current-state)
4. [Target Production Architecture](#4-target-production-architecture)
5. [Phase Overview](#5-phase-overview)
6. [Phase 0 — Project Restructure](#6-phase-0--project-restructure)
7. [Phase 1 — Correctness & Data Integrity](#7-phase-1--correctness--data-integrity)
8. [Phase 2 — Security & API Hardening](#8-phase-2--security--api-hardening)
9. [Phase 3 — Async Processing & Job Queue](#9-phase-3--async-processing--job-queue)
10. [Phase 4 — Infrastructure & Deployment](#10-phase-4--infrastructure--deployment)
11. [Phase 5 — Observability & Quality Assurance](#11-phase-5--observability--quality-assurance)
12. [Phase 6 — Product Features](#12-phase-6--product-features)
13. [Risk Register](#13-risk-register)
14. [Definition of Done — Production MVP](#14-definition-of-done--production-mvp)
15. [Appendix A — Directory Layout](#appendix-a--directory-layout)
16. [Appendix B — Environment Variables](#appendix-b--environment-variables)

---

## 1. Executive Summary

**RAG_XPER** ingests Arabic and English documents — PDFs, scans, images, Markdown, and
plain text — indexes them in a hybrid vector plus lexical store, and answers questions
with Chain-of-Thought reasoning and source attribution.

The pipeline has moved from a local prototype to a deployable service. Phases 0, 1, 3,
and 5 are largely complete; the remaining work is hardening rather than construction.

| Milestone | Status | Outcome |
|-----------|:------:|---------|
| Restructured codebase | ✅ Done | Installable `rag_xper` package, decoupled entry points |
| Correctness fixes | ✅ Done | Content-hash dedup, persisted BM25, shared RRF |
| Deployable service | ✅ Done | Docker Compose, CI, background jobs |
| Observability | ✅ Done | `/health`, `/ready`, `/metrics`, `/version` |
| Security hardening | 🟡 Partial | API keys and upload limits in place; rate limiting pending |
| Durable job queue | ⬜ Open | Jobs are per-process; Redis is provisioned but unused |

---

## 2. Product Vision & Scope

### 2.1 Problem Statement

Organizations handling Arabic legal, regulatory, and administrative documents need a
system that:

- Answers questions **strictly from indexed documents**, with citations
- Handles **scanned content** through Arabic-aware OCR and retrieval
- Supports **article-level chunking** for laws, regulations, and contracts
- Runs **on-premise** when data cannot leave the network

### 2.2 Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│  Document   │────▶│  Ingestion   │────▶│  Retrieval  │────▶│ Generation  │
│ PDF · Image │     │ Extract·OCR  │     │ Hybrid RRF  │     │ CoT + LLM   │
│  MD · TXT   │     │ Chunk·Embed  │     │ BM25+Dense  │     │ + Sources   │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
```

### 2.3 Key Differentiators

| Capability | Description | Primary use case |
|------------|-------------|------------------|
| Arabic OCR pipeline | EasyOCR/PaddleOCR with page rasterization | Scanned Arabic PDFs |
| Article-based chunking | Splits on `المادة`, `Article`, `Section` | Legal and regulatory text |
| Parent-child chunking | Small chunks for search, full parent for context | Large documents |
| Hybrid search (RRF) | Arabic-normalized BM25 fused with dense vectors | Keyword plus semantic match |
| Chain-of-Thought | Reasoning and Answer sections with OCR correction | Auditable answers |
| Folder ingestion | Index files staged on disk, no upload required | Bulk loads on a server |

### 2.4 Out of Scope

Multi-tenant billing, streaming UI, fine-tuned embedding models, and mobile clients.

---

## 3. Current State

### 3.1 Delivered

- `src/rag_xper/` installable package with `pyproject.toml` and console scripts
- `bootstrap.build_orchestrator()` as the single wiring point for CLI, API, and UI
- Content-hash deduplication and a persisted BM25 index
- Shared RRF fusion in `core/retrieval/hybrid_fusion.py`
- API key authentication, CORS configuration, and upload size limits
- Background ingestion jobs with progress tracking (`core/jobs.py`)
- Folder ingestion from `DOCUMENTS_DIR` for server-side document loads
- Observability endpoints and a Docker Compose stack with Qdrant and Redis
- 41 tests covering chunking, retrieval, orchestration, API security, and edge cases

### 3.2 Open Gaps

| ID | Issue | Severity | Impact |
|----|-------|----------|--------|
| G-01 | Auth is optional — empty `API_KEYS` leaves endpoints public | High | Accidental public exposure |
| G-02 | No rate limiting | Medium | Abuse and cost overrun |
| G-03 | Job state is per-process | Medium | Blocks horizontal scaling of the API |
| G-04 | Redis provisioned but unused | Low | Misleading infrastructure |
| G-05 | `/v1/documents` derives counts from the BM25 index | Low | Inaccurate if BM25 and the vector store drift |
| G-06 | Compose publishes 8000, 6333, and 6379 on all interfaces | Medium | Must be restricted before public deployment |
| G-07 | No structured JSON logging despite `LOG_FORMAT=json` in the image | Low | Harder log aggregation |

---

## 4. Target Production Architecture

```
                    ┌─────────────────────────────────────────┐
                    │            nginx / ALB (TLS)             │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │        FastAPI (rag_xper.api.app)        │
                    │   Auth  ·  Validation  ·  Rate limit     │
                    └──────────┬──────────────┬────────────────┘
                               │              │
              ┌────────────────▼──┐    ┌──────▼──────┐
              │  RAGOrchestrator  │    │  Job queue  │
              │   (bootstrap.py)  │    │   (Redis)   │
              └────────┬──────────┘    └─────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ Ingestion│  │ Retrieval│  │Generation│
   │Extract   │  │ Qdrant   │  │ Gemini / │
   │OCR·Chunk │  │ BM25·RRF │  │ Ollama   │
   └──────────┘  └──────────┘  └──────────┘
```

### 4.1 Design Principles

1. **Single wiring point** — components are assembled only in `bootstrap.py`
2. **Async ingestion, synchronous query** — ingestion is slow, queries must stay fast
3. **Config-driven** — every tunable comes from the environment
4. **Fail closed** — authentication, validation, and configuration errors reject requests
5. **Batch resilience** — one bad file never aborts a folder load

---

## 5. Phase Overview

| Phase | Name | Status | Dependencies |
|-------|------|:------:|--------------|
| 0 | Project Restructure | ✅ Done | — |
| 1 | Correctness & Data Integrity | ✅ Done | Phase 0 |
| 2 | Security & API Hardening | 🟡 Partial | Phase 0 |
| 3 | Async Processing & Job Queue | 🟡 Partial | Phases 1, 2 |
| 4 | Infrastructure & Deployment | ✅ Done | Phase 3 |
| 5 | Observability & QA | 🟡 Partial | Phase 4 |
| 6 | Product Features | ⬜ Open | Phase 5 |

**MVP gate:** Phases 0–4 complete. Remaining Phase 2 items are required before exposing
the service on a public network.

---

## 6. Phase 0 — Project Restructure

**Goal:** an installable, maintainable Python package.
**Priority:** P0 · **Status:** ✅ Done

- [x] **0.1** Create `src/rag_xper/` layout
- [x] **0.2** Add `pyproject.toml` with optional dependency extras
- [x] **0.3** Move `core/`, `config.py`, and `utils/` into the package
- [x] **0.4** Add `bootstrap.py` and remove the API's dependency on the CLI
- [x] **0.5** Register console scripts `rag-xper` and `rag-xper-api`
- [x] **0.6** Split optional extras: `ocr-easy`, `ocr-paddle`, `ui`, `dev`, `all`
- [x] **0.7** Update `conftest.py` and test imports for the `src/` layout
- [x] **0.8** Make the test suite hermetic (no `.env` required)

---

## 7. Phase 1 — Correctness & Data Integrity

**Goal:** remove data-layer bugs that cause wrong behaviour in production.
**Priority:** P0 · **Status:** ✅ Done

- [x] **1.1** Replace name-only deduplication with content hashing
- [x] **1.2** Derive the embedding dimension from the provider (`EMBEDDING_DIM`)
- [x] **1.3** Persist the BM25 index so it survives restarts
- [x] **1.4** Extract shared RRF fusion into `core/retrieval/hybrid_fusion.py`
- [x] **1.5** Remove the unused `MMR_LAMBDA` setting
- [x] **1.6** Add Markdown and plain-text ingestion
- [x] **1.7** Support Qdrant server mode through `QDRANT_URL`

### Remaining

- [ ] **1.8** Store `parent_text` once per parent instead of on every child chunk
- [ ] **1.9** Validate the collection dimension on startup and fail with a migration hint

---

## 8. Phase 2 — Security & API Hardening

**Goal:** make the API safe to expose beyond localhost.
**Priority:** P0 · **Status:** 🟡 Partial

- [x] **2.1** `X-API-Key` authentication via `API_KEYS`
- [x] **2.2** Configurable CORS origins
- [x] **2.3** File extension whitelist on upload
- [x] **2.4** Enforce `MAX_UPLOAD_SIZE_MB` while streaming uploads
- [x] **2.5** Path traversal guard on folder ingestion (`DOCUMENTS_DIR` containment)
- [x] **2.6** Sanitize filenames on document deletion
- [x] **2.7** API versioning under `/v1`

### Remaining

- [ ] **2.8** Add `REQUIRE_AUTH` so an empty `API_KEYS` fails closed instead of open
- [ ] **2.9** Rate limiting per API key (`slowapi` or nginx `limit_req`)
- [ ] **2.10** Constant-time key comparison (`secrets.compare_digest`)
- [ ] **2.11** Security headers middleware
- [ ] **2.12** Structured error responses with stable error codes
- [ ] **2.13** Bind Compose ports to `127.0.0.1` and stop publishing 6333 and 6379
- [ ] **2.14** Document the API key rotation procedure

---

## 9. Phase 3 — Async Processing & Job Queue

**Goal:** non-blocking ingestion for large documents and bulk folder loads.
**Priority:** P1 · **Status:** 🟡 Partial

- [x] **3.1** Job model with status, progress, and error tracking
- [x] **3.2** `POST /v1/ingest/async` returning `202` with a `job_id`
- [x] **3.3** `GET /v1/jobs/{job_id}` for progress polling
- [x] **3.4** `POST /v1/ingest/folder` for server-side folder loads
- [x] **3.5** Per-file reporting so one failure cannot abort a batch
- [x] **3.6** Thread-safe job registry
- [x] **3.7** `GET /v1/documents` and `DELETE /v1/documents/{filename}`

### Remaining

- [ ] **3.8** Move job state to Redis so the API can scale beyond one replica
- [ ] **3.9** Run the ingestion worker as a separate container
- [ ] **3.10** Offload blocking orchestrator calls to a thread pool
- [ ] **3.11** Job cancellation and dead-letter handling
- [ ] **3.12** `POST /v1/documents/{filename}/reindex`

---

## 10. Phase 4 — Infrastructure & Deployment

**Goal:** repeatable deployment with a CI pipeline.
**Priority:** P1 · **Status:** ✅ Done

- [x] **4.1** Multi-stage `docker/Dockerfile`
- [x] **4.2** `docker/docker-compose.yml` with API, Qdrant, and Redis
- [x] **4.3** `.dockerignore` excluding secrets, storage, and tests
- [x] **4.4** Qdrant server mode in Compose
- [x] **4.5** GitHub Actions CI running lint and tests
- [x] **4.6** Bind-mount `data/documents` for server-side ingestion
- [x] **4.7** AWS EC2 deployment guide

### Remaining

- [ ] **4.8** `docker-compose.prod.yml` with resource limits and restart policies
- [ ] **4.9** Non-root container user
- [ ] **4.10** Container `HEALTHCHECK` instruction
- [ ] **4.11** Push tagged images to a registry on release
- [ ] **4.12** Automated Qdrant volume backup to S3

---

## 11. Phase 5 — Observability & Quality Assurance

**Goal:** production visibility and confidence in every deploy.
**Priority:** P1 · **Status:** 🟡 Partial

- [x] **5.1** `/health` liveness probe
- [x] **5.2** `/ready` readiness probe
- [x] **5.3** `/metrics` with uptime, counters, and index size
- [x] **5.4** `/version` endpoint
- [x] **5.5** Stress test suite covering diacritics, bulk indexing, and security
- [x] **5.6** API contract tests including folder ingestion and the path guard

### Remaining

- [ ] **5.7** Structured JSON logging with request IDs
- [ ] **5.8** Prometheus exposition format for `/metrics`
- [ ] **5.9** Optional Sentry integration
- [ ] **5.10** Coverage reporting with an 80% target on `core/` and `api/`
- [ ] **5.11** Load test baseline documented in `docs/PERFORMANCE.md`
- [ ] **5.12** End-to-end test against a real sample PDF

---

## 12. Phase 6 — Product Features

**Goal:** enterprise features for multi-user deployments.
**Priority:** P2 · **Status:** ⬜ Open

- [ ] **6.1** Multi-tenancy with per-tenant collections and key mapping
- [ ] **6.2** Streaming answers over Server-Sent Events
- [ ] **6.3** Admin dashboard for documents, jobs, and metrics
- [ ] **6.4** Feedback endpoint and retrieval quality analysis
- [ ] **6.5** Migrate the Gradio UI and CLI to call the REST API

---

## 13. Risk Register

| ID | Risk | Probability | Impact | Mitigation |
|----|------|-------------|--------|------------|
| R-01 | OCR dependencies fail on the target OS | Medium | High | Optional extras; Docker standardizes the environment |
| R-02 | Embedding API rate limits during bulk ingest | High | Medium | Background jobs with retry and batching |
| R-03 | Embedded to server Qdrant migration loses data | Low | High | Back up the volume before switching |
| R-04 | Arabic BM25 quality insufficient for the domain | Medium | Medium | Tunable alpha, feedback loop, optional re-ranker |
| R-05 | LLM hallucination despite Chain-of-Thought | Medium | High | Strict prompt, explicit not-found fallback, required citations |
| R-06 | Large PDF ingestion exhausts memory | Medium | High | Page-by-page processing and worker memory limits |
| R-07 | Service deployed with empty `API_KEYS` | Medium | High | Implement task 2.8 so it fails closed |

---

## 14. Definition of Done — Production MVP

### Code & Structure
- [x] Installable package with decoupled entry points
- [x] Optional dependency extras

### Correctness
- [x] Content-hash deduplication at any collection size
- [x] Provider-aware embedding dimensions
- [x] BM25 index persisted across restarts

### Security
- [x] API key authentication on mutating endpoints
- [x] Upload type and size validation
- [x] Folder ingestion confined to `DOCUMENTS_DIR`
- [ ] Fail closed when no keys are configured
- [ ] Rate limiting enabled

### Operations
- [x] Docker Compose deploys the full stack
- [x] CI passes on every pull request
- [x] Background ingestion with job tracking
- [x] Health, readiness, metrics, and version endpoints

### Quality
- [x] Automated tests for chunking, retrieval, orchestration, and the API
- [ ] Coverage reporting at 80% or above
- [ ] Load test baseline documented

---

## Appendix A — Directory Layout

```
RAG_XPER/
├── src/rag_xper/
│   ├── config.py                 # Settings dataclass
│   ├── bootstrap.py              # build_orchestrator()
│   ├── core/
│   │   ├── ingestion/            # document_extractor · ocr_engine · text_chunker
│   │   ├── retrieval/            # base_vector_store · bm25 · hybrid_fusion · qdrant
│   │   ├── generation/           # llm_interface · rag_orchestrator
│   │   ├── jobs.py               # Background job registry
│   │   ├── models.py
│   │   └── exceptions.py
│   ├── api/app.py                # FastAPI application
│   ├── cli/main.py               # Console entry point
│   └── utils/logger.py
├── apps/gradio_ui/app.py         # Optional demo UI
├── data/documents/               # Server-side ingestion folder (gitignored)
├── docker/                       # Dockerfile · docker-compose.yml
├── docs/                         # PRODUCTION_PLAN.md · DEPLOYMENT_AWS.md
├── tests/                        # Unit, contract, and stress tests
├── .github/workflows/ci.yml
└── pyproject.toml
```

---

## Appendix B — Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | Yes | `gemini` | `gemini` or `ollama` |
| `GEMINI_API_KEY` | If gemini | — | Google AI API key |
| `GEMINI_MODEL` | No | `gemini-3.5-flash-lite` | Generation model |
| `GEMINI_EMBEDDING_MODEL` | No | `gemini-embedding-001` | Embedding model |
| `EMBEDDING_DIM` | No | provider default | 3072 for Gemini, 768 for Ollama |
| `OLLAMA_BASE_URL` | If ollama | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | No | `llama3.1` | Local generation model |
| `OLLAMA_EMBEDDING_MODEL` | No | `nomic-embed-text` | Local embedding model |
| `OCR_ENGINE` | No | `easyocr` | `easyocr` or `paddleocr` |
| `OCR_LANGUAGES` | No | `en,ar` | OCR language codes |
| `VECTOR_STORE_TYPE` | No | `qdrant` | `qdrant` or `chromadb` |
| `QDRANT_URL` | No | — | Server URL; empty means embedded mode |
| `QDRANT_STORAGE_PATH` | No | `./storage/qdrant_db` | Embedded storage path |
| `COLLECTION_NAME` | No | `rag_xper_documents` | Vector collection name |
| `CHUNKING_STRATEGY` | No | `recursive` | `recursive`, `parent_child`, `article_based`, `auto` |
| `USE_HYBRID_SEARCH` | No | `true` | Enable BM25 plus dense fusion |
| `HYBRID_ALPHA` | No | `0.5` | Dense weight in RRF |
| `TOP_K` | No | `6` | Chunks passed to the LLM |
| `FETCH_K` | No | `25` | Candidates fetched before fusion |
| `API_KEYS` | Prod: yes | — | Comma-separated keys for `X-API-Key` |
| `CORS_ORIGINS` | No | `*` | Allowed origins |
| `MAX_UPLOAD_SIZE_MB` | No | `50` | Maximum upload size |
| `DOCUMENTS_DIR` | No | `./data/documents` | Folder scanned by `/v1/ingest/folder` |
| `LOG_LEVEL` | No | `INFO` | Logging level |
