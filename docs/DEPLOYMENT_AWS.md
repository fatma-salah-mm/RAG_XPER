# Deploying RAG_XPER on AWS EC2

This guide covers the **documents-on-disk** deployment: PDFs and images are staged in a
folder on the instance, the API indexes them into Qdrant, and clients query over HTTPS.
Nothing has to be uploaded through the API.

```
        ┌──────────────┐
Client ─┤ nginx / ALB  ├─ TLS termination
        └──────┬───────┘
               │ http://127.0.0.1:8000
        ┌──────▼───────┐     ┌──────────┐     ┌────────┐
        │ rag_xper_api │────▶│  qdrant  │     │ redis  │
        └──────┬───────┘     └────┬─────┘     └────────┘
               │                  │
     /app/data/documents    qdrant_storage volume
        (read-only mount)
```

---

## 1. Prerequisites

| Item | Recommendation |
|------|----------------|
| Instance type | `t3.large` minimum. OCR needs roughly 4 GB RAM; use `t3.xlarge` for large batches. |
| Storage | 40 GB+ gp3 EBS. The image with OCR weights is several GB. |
| OS | Ubuntu 22.04 LTS |
| Security group | Inbound `443` from your users and `22` from your admin IP. **Do not expose 8000, 6333, or 6379 publicly.** |
| Secrets | `GEMINI_API_KEY` and `API_KEYS` — keep them in AWS Secrets Manager or SSM Parameter Store. |

---

## 2. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo tee /etc/apt/keyrings/docker.asc > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

---

## 3. Clone and configure

```bash
sudo mkdir -p /opt/rag && sudo chown $USER:$USER /opt/rag
cd /opt/rag
git clone https://github.com/xper-erp/rag.git .
cp .env.example .env
```

Edit `.env`:

```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=<from Secrets Manager>

VECTOR_STORE_TYPE=qdrant
COLLECTION_NAME=rag_xper_documents
EMBEDDING_DIM=                       # blank derives from the provider

API_KEYS=<openssl rand -hex 32>      # comma-separated for multiple clients
CORS_ORIGINS=https://your-frontend.example.com
MAX_UPLOAD_SIZE_MB=50

LOG_LEVEL=INFO
```

Generate a key:

```bash
openssl rand -hex 32
```

> `QDRANT_URL` and `DOCUMENTS_DIR` are set by `docker-compose.yml`; leave them blank in `.env`.
> `.env` is gitignored — never commit it and never bake it into the image.

---

## 4. Start the stack

```bash
cd /opt/rag/docker
docker compose --env-file ../.env up -d --build
docker compose ps
docker compose logs -f api
```

The first build downloads PyMuPDF and OCR dependencies, so allow 10–20 minutes.

Verify:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

`/ready` returns 503 until Qdrant is reachable and the configuration validates.

---

## 5. Load documents from the folder

Stage files on the host. The folder is bind-mounted read-only into the container:

```bash
cp /path/to/*.pdf /opt/rag/data/documents/
```

Trigger indexing:

```bash
export KEY=<your API key>

curl -X POST http://127.0.0.1:8000/v1/ingest/folder \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{"strategy": "auto", "recursive": true}'
```

The response is `202 Accepted` with a `job_id`. Poll it:

```bash
curl -H "X-API-Key: $KEY" http://127.0.0.1:8000/v1/jobs/<job_id>
```

When the job completes, `details` contains the per-file breakdown:

```json
{
  "status": "completed",
  "progress": 100,
  "chunks_ingested": 412,
  "details": {
    "ingested": 12,
    "skipped": 3,
    "failed": 1,
    "files": [
      {"file": "contract.pdf", "status": "ingested", "chunks": 58},
      {"file": "scan.pdf", "status": "failed", "chunks": 0, "error": "..."}
    ]
  }
}
```

A failing file is reported, not fatal — the rest of the batch still completes.

Confirm what is indexed:

```bash
curl -H "X-API-Key: $KEY" http://127.0.0.1:8000/v1/documents
```

Files already indexed are skipped by content hash, so re-running the job after adding
new documents only processes the new ones. Pass `{"force": true}` to re-index everything.

### Alternative: index from inside the container

```bash
docker compose exec api rag-xper ingest-dir --recursive --strategy auto
```

---

## 6. Query

```bash
curl -X POST http://127.0.0.1:8000/v1/ask \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي شروط الإنهاء في المادة الخامسة؟", "top_k": 6}'
```

The response contains `answer`, `reasoning`, and `sources` with file name, page, and score.

Interactive docs live at `/docs` — reach them over an SSH tunnel rather than exposing the port.

---

## 7. Put TLS in front

Terminate TLS with nginx on the instance or an Application Load Balancer.

```nginx
server {
    listen 443 ssl;
    server_name rag.example.com;

    ssl_certificate     /etc/letsencrypt/live/rag.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/rag.example.com/privkey.pem;

    client_max_body_size 60M;   # keep above MAX_UPLOAD_SIZE_MB

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 300s;   # ingestion and LLM calls are slow
    }
}
```

With an ALB, point the target group health check at `/health`.

> `docker-compose.yml` publishes port 8000 on all interfaces. For a public instance,
> change the mapping to `127.0.0.1:8000:8000` so only the reverse proxy can reach it,
> and drop the published `6333`/`6379` ports.

---

## 8. Operations

### Update to a new release

```bash
cd /opt/rag && git pull
cd docker && docker compose --env-file ../.env up -d --build
```

### Logs and metrics

```bash
docker compose logs -f api
curl http://127.0.0.1:8000/metrics
```

`/metrics` reports uptime, query and ingest counters, error count, and the number of
chunks in the BM25 index.

### Backup

Vector data lives in the `qdrant_storage` Docker volume:

```bash
docker volume ls   # confirm the exact name first
docker run --rm \
  -v docker_qdrant_storage:/data \
  -v /opt/rag/backups:/backup \
  alpine tar czf /backup/qdrant-$(date +%F).tar.gz -C /data .
```

Sync backups to S3 on a schedule.

### Remove a document

```bash
curl -X DELETE -H "X-API-Key: $KEY" \
  http://127.0.0.1:8000/v1/documents/contract.pdf
```

### Rotate API keys

Add the new key alongside the old one in `API_KEYS`, restart, migrate clients, then
remove the old key and restart again:

```bash
docker compose --env-file ../.env up -d api
```

---

## 9. Endpoint reference

| Method | Path | Auth | Purpose |
|--------|------|:----:|---------|
| GET | `/health` | — | Liveness probe |
| GET | `/ready` | — | Readiness probe |
| GET | `/metrics` | — | Counters and index size |
| GET | `/version` | — | Application version |
| POST | `/v1/ingest` | ✅ | Upload and index one file (synchronous) |
| POST | `/v1/ingest/async` | ✅ | Upload and index one file (returns `job_id`) |
| POST | `/v1/ingest/folder` | ✅ | Index files staged under `DOCUMENTS_DIR` (returns `job_id`) |
| GET | `/v1/jobs/{job_id}` | ✅ | Job progress and per-file report |
| POST | `/v1/ask` | ✅ | Ask a question |
| GET | `/v1/documents` | ✅ | List indexed files with chunk counts |
| DELETE | `/v1/documents/{filename}` | ✅ | Delete all chunks for a file |

Authentication is enforced only when `API_KEYS` is set. Leaving it empty makes every
endpoint public — always set it before exposing the instance.

---

## 10. Known limits

| Limit | Detail | Workaround |
|-------|--------|------------|
| Jobs are per process | Job state lives in the API process memory | Keep the API at one replica until jobs move to Redis |
| Embedding dimension is fixed per collection | Switching Gemini ↔ Ollama needs a new collection | Set a new `COLLECTION_NAME` and re-ingest |
| No rate limiting | Not yet implemented | Apply `limit_req` in nginx or an ALB/WAF rule |
| Redis is provisioned but unused | Reserved for the durable job queue | No action needed today |

See [`PRODUCTION_PLAN.md`](./PRODUCTION_PLAN.md) for the roadmap addressing these.
