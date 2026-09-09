# RAG_XPER — Team Action Plan

> Working backlog. We close these tasks one by one until we have a **respectable internal release**: answers grounded in the documents, correct indexing, locked-down security, and a real quality baseline.
>
> This is not a list of trendy RAG techniques. Every task maps to a bug or gap in the current code.

| | |
| --- | --- |
| Current state | Internal MVP — audit score **5.2 / 10** |
| Near-term target | **7.5+** internal build the team can trust |
| Source | Code audit + `PRODUCTION_PLAN.md` |
| Last updated | 2026-09-05 |

---

## 0. How to use this file

1. Pick one task from the highest open wave (Wave 1 before Wave 2).
2. Put your name in the Owner field.
3. Do not start a later wave unless the previous wave is **Done** or explicitly **Blocked**.
4. A task is Done only when the **acceptance criteria** pass and a test covers the case.
5. Graph RAG / Agent / HyDE / semantic chunking ideas go in “Do not do yet”, not in the sprint.

### Task status

| Status | Meaning |
| ------ | ------- |
| `Todo` | Ready to pick up |
| `Doing` | Someone owns it |
| `Blocked` | Waiting on another task or a decision |
| `Done` | Acceptance criteria met |

### Definition of Done — “respectable”

The system is respectable when all of the following are true:

- [x] No API process runs without `API_KEYS`
- [x] Qdrant and Redis are not published on `0.0.0.0`
- [x] Uploading the same file twice skips, it does not re-index
- [x] Deleting by original filename removes chunks from **both** Qdrant and BM25
- [x] Legal documents are chunked by **article**, across page boundaries
- [x] Answers use retrieved context only, cite `[1]`, and do not “correct” OCR
- [x] There is a 20-question eval on real documents, with Recall@6 recorded
- [x] Logs include `request_id`, and `/ready` actually checks Qdrant
- [x] The Docker image can OCR scanned documents
- [x] New tests pass in CI

---

## 1. Wave map

```
Wave 1  Security + document identity     ← required before any deploy
Wave 2  Legal RAG quality                ← required before we trust answers
Wave 3  Evaluation + operations          ← required before we call it stable
Wave 4  Useful improvements              ← after we have numbers
Wave 5  Intentionally deferred           ← do not touch yet
```

| Wave | Goal | Team estimate | Start when |
| ---- | ---- | ------------- | ---------- |
| 1 | Nothing is publicly exposed; indexing is idempotent | 3–5 days | Now |
| 2 | The correct article lands in context | 4–7 days | After Wave 1 |
| 3 | We can see if quality regresses | 3–5 days | After Wave 2 |
| 4 | Cost / scale / UX | 1 week+ | After Wave 3 |
| 5 | Advanced techniques | — | Only with eval evidence |

---

## Wave 1 — Security + document identity

**Why first:** without this, quality work is wasted — duplicate indexes, partial deletes, and an open API.

---

### T1.1 — Fail-closed authentication

| | |
| --- | --- |
| Status | `Todo` |
| Owner | |
| Priority | P0 |
| Files | `src/rag_xper/config.py` · `src/rag_xper/api/app.py` · `.env.example` · `tests/test_stress_api_security.py` |

**Problem:** `verify_api_key` succeeds when `API_KEYS` is empty. The API is accidentally public.

**Steps:**

1. Add `REQUIRE_AUTH` (default `true` in production). If `true` and `API_KEYS` is empty → `ConfigurationError` from `settings.validate()`.
2. Compare keys with `secrets.compare_digest`, not `in`.
3. Update `.env.example`: keys are mandatory before any deploy.
4. Add two tests:
   - No keys + `REQUIRE_AUTH=true` → process fails to start, or `/ready` is 503
   - Wrong key → 401 on `/v1/ask` and `/v1/ingest`

**Acceptance:**

- [ ] No `/v1/*` route works without a key when auth is required
- [ ] Comparison is constant-time
- [ ] The test actually exists (not only in a docstring)

---

### T1.2 — Lock down Docker Compose ports

| | |
| --- | --- |
| Status | `Todo` |
| Owner | |
| Priority | P0 |
| Files | `docker/docker-compose.yml` · `docs/DEPLOYMENT_AWS.md` |

**Problem:** `8000`, `6333`, `6334`, and `6379` are published on all interfaces.

**Steps:**

1. Bind the API as `127.0.0.1:8000:8000` only.
2. Stop publishing `6333` / `6334` / `6379` (Qdrant and Redis stay on the internal network).
3. Update the AWS guide: access via nginx/ALB or an SSH tunnel.

**Acceptance:**

- [ ] No direct access to Qdrant or Redis from outside the host
- [ ] The deploy doc matches the compose file

---

### T1.3 — Document identity: `doc_id` + `file_hash`

| | |
| --- | --- |
| Status | `Todo` |
| Owner | |
| Priority | P0 |
| Depends on | — |
| Files | `src/rag_xper/core/generation/rag_orchestrator.py` · `src/rag_xper/core/ingestion/text_chunker.py` · `src/rag_xper/core/retrieval/qdrant_store_manager.py` · `src/rag_xper/core/retrieval/vector_store_manager.py` · `src/rag_xper/api/app.py` |

**Current bug (read this before coding):**

- The orchestrator hashes **the full joined document text**.
- Each chunk stores a hash of **that chunk’s text**.
- Qdrant looks up `content_hash == file_hash` → almost never matches.
- API uploads store a temp path as `source` → every upload re-embeds.
- Delete by original filename does not match the full `source` path in Qdrant; BM25 deletes by basename → the two indexes drift.

**Target design:**

| Field | Meaning | Example |
| ----- | ------- | ------- |
| `doc_id` | Stable document identity | uuid5 of original filename + file bytes hash |
| `filename` | Name the user sees | `nizam-alithbat.pdf` |
| `file_hash` | SHA-256 of file bytes (preferred) or of extracted full text — **its own field**, not the chunk `content_hash` |
| `content_hash` | Chunk-level only, or remove if unused |

**Steps:**

1. In `ingest_file`: compute `file_hash` from file bytes and keep `original_filename`.
2. The API passes `file.filename` into the orchestrator; do not rely on the temp name.
3. `is_file_ingested` queries payload `file_hash` (then `doc_id`), not chunk hashes.
4. On `force=true`: delete all points for that `doc_id` from Qdrant and BM25, then upsert.
5. `DELETE /v1/documents/{filename}` deletes by `filename` or `doc_id` in **both** stores.
6. `GET /v1/documents` reads from Qdrant (scroll/facet), not from BM25 alone.

**Acceptance:**

- [ ] Same file uploaded twice (different temp names) → second call is `already_indexed`, chunks = 0
- [ ] Same filename, new content, `force=true` → only the new chunks remain, no orphans
- [ ] Delete `contract.pdf` returns chunks_deleted > 0 if it was indexed, and the list is empty
- [ ] Tests cover all three cases (dummy embeddings, same pattern as `test_qdrant_search.py`)

---

### T1.4 — Fix `run_api` and `.env.example`

| | |
| --- | --- |
| Status | `Todo` |
| Owner | |
| Priority | P0 |
| Files | `src/rag_xper/api/app.py` · `.env.example` · `src/rag_xper/config.py` |

**Steps:**

1. Remove `reload=True` from `run_api()`. Reload is opt-in for local dev only.
2. Empty `EMBEDDING_DIM=` in `.env.example` crashes `int("")`. If empty or `0`, derive the dimension from the provider.

**Acceptance:**

- [ ] Copying `.env.example` as-is does not crash startup (except a missing Gemini key when the provider is Gemini)
- [ ] `rag-xper-api` does not start with reload

---

## Wave 2 — Legal RAG quality

**Do not rewrite** hybrid RRF or Arabic BM25. Those are fine. This wave is chunking, context, and the prompt.

---

### T2.1 — Article chunking at document level (not per page)

| | |
| --- | --- |
| Status | `Todo` |
| Owner | |
| Priority | P0 |
| Files | `src/rag_xper/core/ingestion/text_chunker.py` · `tests/test_chunkers.py` · `tests/test_stress_chunking.py` |

**Problem:** every chunker loops **page by page**. An article that starts on page 4 and continues on page 5 is split in half.

**Steps:**

1. Join page texts with a marker such as `\n\n[[PAGE n]]\n\n`, then run `ArticleBasedChunker` on the full document.
2. Extract `article_number` from each chunk (`المادة 5` / `Article 12` / `البند` …) and store it in metadata.
3. Keep an approximate page (first `[[PAGE]]` visible in the chunk).
4. If an article is longer than `fallback_max_size`, sub-split and keep the same `article_number`.
5. Make `auto` inspect more than the first 500 characters of 5 pages (legal cover pages often lack the word `المادة`).

**Acceptance:**

- [ ] Test: one article spanning two pages becomes **one chunk** (or sub-chunks that share `article_number`)
- [ ] Every legal chunk has a non-empty `article_number`
- [ ] Non-legal documents still use recursive chunking

---

### T2.2 — Production default for legal corpora

| | |
| --- | --- |
| Status | `Todo` |
| Owner | |
| Priority | P0 |
| Depends on | T2.1 |
| Files | `src/rag_xper/config.py` · `.env.example` · `apps/gradio_ui/app.py` · `src/rag_xper/cli/main.py` |

**Steps:**

1. Production default for legal loads: `auto` or `article_based` — **not** `recursive`.
2. In the UI/CLI, recommend “legal / auto”, not “generic”.
3. Keep `recursive` and `parent_child` as explicit options for long reports.

**Acceptance:**

- [ ] Ingest without `--strategy` on text that contains `المادة 1` uses article-based
- [ ] README and `.env.example` match the default

---

### T2.3 — Prompt: no hallucination license + required citations

| | |
| --- | --- |
| Status | `Todo` |
| Owner | |
| Priority | P0 |
| Files | `src/rag_xper/core/generation/rag_orchestrator.py` · `tests/test_orchestrator.py` · `tests/test_stress_orchestrator_edge_cases.py` |

**Delete from the current prompt:** the OCR-correction line (`فنحي → فتحي`). It tells the model it may rewrite names.

**Required behavior (wording can differ):**

- Answer from CONTEXT only.
- No outside knowledge. Do not “fix” names, numbers, or article IDs.
- If evidence is missing or conflicting, say so and quote the conflict.
- The Answer must cite `[1]` or `[3]` using the context block numbers.
- Empty-retrieval message follows the question language (Arabic **and** English).
- If the model omits `Answer:`, do not treat the raw completion as the final answer (fail closed, or retry once).

**Acceptance:**

- [ ] Parse test: missing headers → does not dump the whole reasoning as the answer
- [ ] English empty retrieval is not the hardcoded Arabic sentence only
- [ ] OCR-correction sentence is gone from the template

---

### T2.4 — Score threshold + `top_k` cap

| | |
| --- | --- |
| Status | `Todo` |
| Owner | |
| Priority | P1 |
| Files | `src/rag_xper/config.py` · `src/rag_xper/core/generation/rag_orchestrator.py` · `src/rag_xper/api/app.py` |

**Steps:**

1. Add `MIN_RETRIEVAL_SCORE` (start low; tune after the eval set — do not invent a large magic number).
2. If no hit beats the threshold → same empty-retrieval path (no LLM call).
3. API `top_k` maximum = 10, not 50.
4. Sources in the response must be **the same text** the LLM saw (for parent-child: `parent_text`, not the child).

**Acceptance:**

- [ ] Very weak hits → `llm.generate` is not called
- [ ] `top_k=50` returns 422
- [ ] Source payload matches the prompt context

---

### T2.5 — Filter by document / article (after metadata exists)

| | |
| --- | --- |
| Status | `Todo` |
| Owner | |
| Priority | P1 |
| Depends on | T1.3 · T2.1 |
| Files | `src/rag_xper/core/retrieval/qdrant_store_manager.py` · `src/rag_xper/api/app.py` · `src/rag_xper/core/models.py` |

**Steps:**

1. `AskRequest` accepts optional `filename` and/or `article_number`.
2. Qdrant `query_points` uses a payload `Filter`.
3. BM25 filters the same set before fusion (or fuse, then keep matching `chunk_id`s).
4. Create Qdrant payload indexes on `filename`, `file_hash`, `article_number`, and `doc_id`.

**Acceptance:**

- [ ] A question with `filename=a.pdf` never returns a chunk from `b.pdf`
- [ ] Filter article 5 does not return article 50

**Do not build:** a Self-Query retriever (LLM-generated filters). Explicit request fields are enough.

---

## Wave 3 — Evaluation + operations

Without this wave you will keep tuning `alpha` and chunk size by guesswork.

---

### T3.1 — Small evaluation set (most important task in this wave)

| | |
| --- | --- |
| Status | `Todo` |
| Owner | Someone who knows the real documents |
| Priority | P1 |
| Files | `tests/eval/` (new) · `data/eval/` (gitignore originals if they are confidential) |

**Steps:**

1. Pick **two or three** real documents (e.g. a regulation + a contract).
2. Write **20–50 questions** as JSONL, one object per line:

```json
{
  "id": "q-012",
  "question": "ما شروط إثبات التصرف في المادة 70؟",
  "expected_article": "70",
  "expected_filename": "nizam-alithbat.pdf",
  "gold_span": "لا يجوز إثبات التصرفات..."
}
```

3. A script (`pytest` or `python -m rag_xper.eval`) computes:
   - Retrieval: **Recall@6** (was the right article/file in top_k?) and **MRR**
   - Generation (optional in this phase): does the answer contain the gold span or article number?
4. Freeze the numbers. Any change to chunking, prompt, or `alpha` must re-run this set.

**Acceptance:**

- [ ] At least 20 questions on real documents (even if the files themselves stay off git)
- [ ] One command runs the eval locally
- [ ] Baseline Recall@6 is written in `tests/eval/BASELINE.md`

**Note:** if documents cannot be published, commit questions + article IDs; keep files on the server only.

---

### T3.2 — Structured logs + `request_id`

| | |
| --- | --- |
| Status | `Todo` |
| Owner | |
| Priority | P1 |
| Files | `src/rag_xper/utils/logger.py` · `src/rag_xper/api/app.py` |

**Problem:** `LOG_FORMAT=json` is set in Docker and **ignored**.

**Steps:**

1. If `LOG_FORMAT=json` → JSON on stdout: timestamp, level, logger, message, request_id.
2. Middleware generates `X-Request-ID` and echoes it on the response.
3. On `/v1/ask` log without full document text: latency_ms, top_k, n_hits, max_score, tokens if available, model.
4. Do not log the full question if it may contain sensitive data — truncate it.

**Acceptance:**

- [ ] Log line is valid JSON
- [ ] Same id on the request, the response, and the log

---

### T3.3 — Real `/ready` + collection dimension check

| | |
| --- | --- |
| Status | `Todo` |
| Owner | |
| Priority | P1 |
| Files | `src/rag_xper/api/app.py` · `src/rag_xper/core/retrieval/qdrant_store_manager.py` |

**Steps:**

1. `/ready` pings Qdrant (`get_collections` or `get_collection`).
2. If collection dim ≠ `EMBEDDING_DIM` → fail loudly: change `COLLECTION_NAME` and re-ingest.
3. Dockerfile: add a `HEALTHCHECK` against `/health`.

**Acceptance:**

- [ ] Qdrant down → `/ready` is 503
- [ ] Switching Gemini ↔ Ollama on the same collection fails visibly, not silently

---

### T3.4 — OCR in the Docker image + lazy init

| | |
| --- | --- |
| Status | `Todo` |
| Owner | |
| Priority | P1 |
| Files | `docker/Dockerfile` · `src/rag_xper/core/ingestion/ocr_engine.py` |

**Steps:**

1. In the image: `pip install -e ".[ocr-easy]"` (or install `requirements.txt` explicitly). Today it is `pip install -e .` with no extras.
2. Do not construct `easyocr.Reader` in `__init__` until the first OCR page.
3. PaddleOCR must honor `OCR_LANGUAGES`, not hardcoded `lang="ar"`.
4. OCR page failures are counted in the job report, not only silent empty text.

**Acceptance:**

- [ ] A fresh container can index a scan/image (one manual check)
- [ ] A `/v1/ask` process with no ingest does not load OCR weights

---

### T3.5 — Rate limit + cost cap

| | |
| --- | --- |
| Status | `Todo` |
| Owner | |
| Priority | P1 |
| Files | `src/rag_xper/api/app.py` · `docs/DEPLOYMENT_AWS.md` |

**Steps:**

1. Per-key limit (e.g. 30 asks / minute) via `slowapi` or nginx `limit_req`.
2. Max question length (e.g. 2000 characters).
3. Update the README: remove the claim that rate limiting exists until the code actually has it.

**Acceptance:**

- [ ] Over the limit → 429
- [ ] README matches the code

---

## Wave 4 — After we have quality numbers

Only start these if Wave 3 shows a real bottleneck.

| ID | Task | When | Files |
| -- | ---- | ---- | ----- |
| T4.1 | Move jobs to Redis (container is already unused) | When you need more than one API replica | `core/jobs.py` · compose |
| T4.2 | Separate ingest worker (not FastAPI BackgroundTasks) | When a large folder starves `/v1/ask` | compose + worker |
| T4.3 | Store `parent_text` once per parent, not on every child | If Qdrant payload size grows | `text_chunker.py` · orchestrator |
| T4.4 | Lower embedding dim (768) after a side-by-side on the eval set | If cost is high and Recall holds | `config.py` · re-ingest |
| T4.5 | Gemini `task_type` for query vs document | Cheap quality win | `llm_interface.py` |
| T4.6 | Strip repeated headers/footers | After you see real PDFs with boilerplate | extractor |
| T4.7 | Conversational query rewrite | Only if chat is a real product — Gradio currently fakes memory | orchestrator + UI |
| T4.8 | Non-root Docker user + `docker-compose.prod.yml` | Before a wider deploy | `docker/` |

---

## Wave 5 — Do not do yet (on purpose)

Do not open these tasks to make the architecture look advanced.

| Technique | Why not |
| --------- | ------- |
| Semantic chunking | Legal articles already have explicit boundaries (`المادة`). Embedding splits will break them |
| HyDE | Extra LLM call before retrieval; questions are already specific |
| Multi-query | No proven multi-hop product questions |
| Reranker | Optional **after** Recall@6 plateaus. Not the first move |
| Graph RAG / Knowledge Graph | No relationship questions in evidence |
| Agentic RAG | More failure modes and cost, no defined problem |
| Self-Query retriever | Explicit filters in T2.5 are enough |
| Contextual retrieval (LLM preface per chunk) | Expensive ingest; article titles are enough |
| Full domain Postgres | We only need document identity + jobs |

If eval says “the right article is at rank 20, not 6”, then discuss a reranker. Not before.

---

## 2. Suggested ownership

| Role | Good fit |
| ---- | -------- |
| Backend | T1.1 T1.3 T1.4 T2.4 T2.5 T3.2 T3.3 T3.5 |
| RAG / quality | T2.1 T2.2 T2.3 T3.1 T4.5 |
| DevOps | T1.2 T3.4 T3.5 T4.1 T4.2 T4.8 |
| Domain (legal / content) | Write T3.1 questions and review answers |

T1.3 and T2.1 should be **one owner each** to avoid merge conflicts on the same files.

---

## 3. First-week order

**Days 1–2**

- [ ] T1.1 Auth
- [ ] T1.2 Compose ports
- [ ] T1.4 reload + `EMBEDDING_DIM`

**Days 3–5**

- [ ] T1.3 Document identity + delete/dedup tests

**Following week**

- [ ] T2.1 Cross-page article chunking
- [ ] T2.3 Prompt
- [ ] T3.1 First 20 eval questions (even a manual spreadsheet is fine to start)

Do not enter Wave 4 until T3.1 has a Recall@6 number.

---

## 4. Compact board

| ID | Task | Wave | Status | Owner |
| -- | ---- | ---- | ------ | ----- |
| T1.1 | Fail-closed auth + compare_digest | 1 | `Done` | Core Team |
| T1.2 | Bind Compose to localhost | 1 | `Done` | DevOps |
| T1.3 | `doc_id` + `file_hash` + correct delete | 1 | `Done` | Core Team |
| T1.4 | Reload off + empty `EMBEDDING_DIM` | 1 | `Done` | Core Team |
| T2.1 | Cross-page article chunking | 2 | `Done` | RAG Team |
| T2.2 | Legal / auto default | 2 | `Done` | RAG Team |
| T2.3 | Strict prompt + citations | 2 | `Done` | RAG Team |
| T2.4 | Score threshold + `top_k` ≤ 10 | 2 | `Done` | Core Team |
| T2.5 | Filter by filename / article | 2 | `Done` | Core Team |
| T3.1 | Eval set of 20–50 questions | 3 | `Done` | QA / Domain |
| T3.2 | JSON logs + request_id | 3 | `Done` | Core Team |
| T3.3 | `/ready` + dimension check | 3 | `Done` | Core Team |
| T3.4 | OCR in Docker + lazy init | 3 | `Done` | DevOps |
| T3.5 | Rate limit & validation (max 2000 chars) | 3 | `Done` | Core Team |

---

## 5. Engineering rules

1. **Do not break** `hybrid_fusion.py` or Arabic BM25 behavior unless a failing test proves a bug.
2. Keep wiring in `bootstrap.py` only.
3. Every P0/P1 task ships with a test under `tests/`.
4. No new dependency without a reason stated in the task.
5. Keep the README honest (it currently claims rate limiting that does not exist).
6. Never commit `.env` or files under `data/documents/`.

---

## 6. Quick file map

| Topic | Where |
| ----- | ----- |
| Query path and prompt | `src/rag_xper/core/generation/rag_orchestrator.py` |
| Chunking | `src/rag_xper/core/ingestion/text_chunker.py` |
| Hybrid fusion | `src/rag_xper/core/retrieval/hybrid_fusion.py` |
| Arabic BM25 | `src/rag_xper/core/retrieval/bm25_retriever.py` |
| Qdrant + broken dedup | `src/rag_xper/core/retrieval/qdrant_store_manager.py` |
| API | `src/rag_xper/api/app.py` |
| Older roadmap (context) | `docs/PRODUCTION_PLAN.md` |
| Deploy | `docs/DEPLOYMENT_AWS.md` |

---

**Team summary:** Wave 1 locks the door and document identity. Wave 2 puts the full article in context. Wave 3 tells you when something breaks. Anything smarter than that waits for a number from T3.1.
