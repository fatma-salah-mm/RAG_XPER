"""
rag_xper.config

Centralised configuration for the RAG_XPER pipeline, loaded from environment
variables with sensible defaults and fast validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_xper.core.exceptions import ConfigurationError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Provider selection ---
    llm_provider: Literal["gemini", "ollama"] = "gemini"

    # --- LLM (Gemini) ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-001"

    # --- Embedding Dimension ---
    embedding_dim: int = 0

    # --- LLM (Ollama) ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_embedding_model: str = "nomic-embed-text"

    # --- LLM (shared) ---
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 3

    # --- OCR ---
    ocr_engine: str = "easyocr"
    ocr_languages: tuple[str, ...] = ("en", "ar")
    native_text_min_chars: int = 20
    ocr_render_zoom: float = 2.5

    # --- Vector Store ---
    vector_store_type: Literal["qdrant", "chromadb"] = "qdrant"
    qdrant_url: str | None = None
    qdrant_storage_path: str = "./storage/qdrant_db"
    vector_db_path: str = "./storage/chroma_db"
    collection_name: str = "rag_xper_documents"

    # --- Chunking ---
    chunking_strategy: str = "auto"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    parent_chunk_size: int = 1500
    child_chunk_size: int = 300

    # --- Retrieval ---
    use_hybrid_search: bool = True
    hybrid_alpha: float = 0.5
    top_k: int = 6
    fetch_k: int = 25
    min_retrieval_score: float = 0.0

    # --- API Security ---
    app_env: Literal["development", "production", "test"] = "development"
    require_auth: bool = False
    api_keys: tuple[str, ...] = ()
    max_upload_size_mb: int = 50
    cors_origins: tuple[str, ...] = ("*",)
    docs_enabled: bool = True
    metrics_require_auth: bool = False
    rate_limit_per_minute: int = 30

    # --- Documents ---
    documents_dir: str = "./data/documents"

    # --- MySQL ---
    mysql_host: str | None = None
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "rag_xper_db"
    mysql_required: bool = False
    run_db_migrations: bool = False

    # --- Redis (optional job backend for multi-replica) ---
    redis_url: str | None = None

    # --- Cache ---
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000

    @field_validator("embedding_dim", mode="before")
    @classmethod
    def _parse_embedding_dim(cls, value: object) -> int:
        if value is None or value == "":
            return 0
        return int(value)

    @field_validator("ocr_languages", "api_keys", "cors_origins", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        if isinstance(value, (list, tuple)):
            return tuple(str(item).strip() for item in value if str(item).strip())
        return ()

    @field_validator("vector_store_type", "chunking_strategy", "app_env", mode="before")
    @classmethod
    def _lowercase(cls, value: object) -> str:
        return str(value).lower() if value is not None else ""

    @field_validator(
        "use_hybrid_search",
        "require_auth",
        "cache_enabled",
        "docs_enabled",
        "metrics_require_auth",
        "mysql_required",
        "run_db_migrations",
        mode="before",
    )
    @classmethod
    def _parse_bool(cls, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    @model_validator(mode="after")
    def _derive_embedding_dim(self) -> Settings:
        if self.embedding_dim <= 0:
            object.__setattr__(
                self,
                "embedding_dim",
                3072 if self.llm_provider == "gemini" else 768,
            )
        return self

    def validate(self) -> None:
        """Fail fast if required configuration is missing or unusable."""
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ConfigurationError("GEMINI_API_KEY is not set in .env. Please set GEMINI_API_KEY.")
        if self.require_auth and not self.api_keys:
            raise ConfigurationError("REQUIRE_AUTH is true but API_KEYS is empty. Please set API_KEYS in .env.")
        if self.app_env == "production":
            if not self.require_auth or not self.api_keys:
                raise ConfigurationError(
                    "Production deployment requires REQUIRE_AUTH=true and non-empty API_KEYS."
                )
            if "*" in self.cors_origins:
                raise ConfigurationError(
                    "Production deployment cannot use CORS_ORIGINS=*; set explicit allowed origins."
                )
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if self.child_chunk_size >= self.parent_chunk_size:
            raise ConfigurationError("CHILD_CHUNK_SIZE must be smaller than PARENT_CHUNK_SIZE.")

        if self.vector_store_type == "qdrant":
            if not self.qdrant_url:
                Path(self.qdrant_storage_path).mkdir(parents=True, exist_ok=True)
        else:
            Path(self.vector_db_path).mkdir(parents=True, exist_ok=True)


settings = Settings()
