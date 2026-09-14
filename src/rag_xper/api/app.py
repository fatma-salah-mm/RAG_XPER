"""
rag_xper.api.app

FastAPI production application for RAG_XPER.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from rag_xper import __version__
from rag_xper.api import state
from rag_xper.api.rate_limit import limiter
from rag_xper.api.routes import ask, documents, health, ingest
from rag_xper.config import settings
from rag_xper.core.db.session import init_db
from rag_xper.core.exceptions import ConfigurationError, RAGPipelineError
from rag_xper.utils.logger import get_logger, reset_request_id, set_request_id

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate configuration at startup and release resources on shutdown."""
    settings.validate()
    logger.info("RAG_XPER API starting (env=%s)", settings.app_env)
    yield
    state.close_resources()
    logger.info("RAG_XPER API shutdown complete")


app = FastAPI(
    title="RAG_XPER Enterprise API",
    version=__version__,
    description=(
        "Production-grade Arabic/English Multi-Modal RAG API with Qdrant, "
        "Persisted BM25, In-Memory Cache, MySQL, and Web UI."
    ),
    lifespan=lifespan,
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
    openapi_url="/openapi.json" if settings.docs_enabled else None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(ConfigurationError)
async def configuration_error_handler(_request: Request, exc: ConfigurationError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(RAGPipelineError)
async def pipeline_error_handler(_request: Request, exc: RAGPipelineError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


try:
    init_db()
except Exception as db_exc:
    logger.warning("Database startup notice: %s", db_exc)

origins = list(settings.cors_origins) if settings.cors_origins else ["*"]
allow_credentials = "*" not in origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = req_id
    token = set_request_id(req_id)
    try:
        response = await call_next(request)
    finally:
        reset_request_id(token)
    response.headers["X-Request-ID"] = req_id
    return response


app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(ask.router)
app.include_router(documents.router)

# Backward-compatible aliases used by tests and legacy imports
from rag_xper.api.state import get_orchestrator  # noqa: E402, F401

ui_dir = Path(__file__).resolve().parent.parent.parent.parent / "apps" / "web_dashboard"
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/ui/")


def run_api() -> None:
    """Console script launcher: rag-xper-api"""
    uvicorn.run("rag_xper.api.app:app", host="0.0.0.0", port=8000)  # nosec B104


if __name__ == "__main__":
    run_api()
