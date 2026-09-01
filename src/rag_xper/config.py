"""
rag_xper.config

Centralised configuration for the RAG_XPER pipeline, loaded from environment
variables with sensible defaults and fast validation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from rag_xper.core.exceptions import ConfigurationError

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # --- Provider selection ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" | "ollama"

    # --- LLM (Gemini) ---
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    # --- Embedding Dimension (Phase 1 auto-parameterization) ---
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "3072" if os.getenv("LLM_PROVIDER", "gemini") == "gemini" else "768"))

    # --- LLM (Ollama -- fully local, no API key) ---
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    # --- LLM (shared across providers) ---
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))

    # --- OCR ---
    ocr_engine: str = os.getenv("OCR_ENGINE", "easyocr")  # "easyocr" | "paddleocr"
    ocr_languages: tuple = tuple(os.getenv("OCR_LANGUAGES", "en,ar").split(","))
    native_text_min_chars: int = int(os.getenv("NATIVE_TEXT_MIN_CHARS", "20"))
    ocr_render_zoom: float = float(os.getenv("OCR_RENDER_ZOOM", "2.5"))

    # --- Vector Store (Qdrant & ChromaDB) ---
    vector_store_type: str = os.getenv("VECTOR_STORE_TYPE", "qdrant").lower()  # "qdrant" | "chromadb"
    qdrant_url: Optional[str] = os.getenv("QDRANT_URL", None)
    qdrant_storage_path: str = os.getenv("QDRANT_STORAGE_PATH", "./storage/qdrant_db")
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "./storage/chroma_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "rag_xper_documents")

    # --- Modular Chunking Strategy ---
    chunking_strategy: str = os.getenv("CHUNKING_STRATEGY", "recursive").lower()
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    parent_chunk_size: int = int(os.getenv("PARENT_CHUNK_SIZE", "1500"))
    child_chunk_size: int = int(os.getenv("CHILD_CHUNK_SIZE", "300"))

    # --- Retrieval ---
    use_hybrid_search: bool = os.getenv("USE_HYBRID_SEARCH", "true").lower() in ("true", "1", "yes")
    hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "0.5"))
    top_k: int = int(os.getenv("TOP_K", "6"))
    fetch_k: int = int(os.getenv("FETCH_K", "25"))

    # --- API Security (Phase 2) ---
    api_keys: tuple = tuple([k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()])
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    cors_origins: tuple = tuple([o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()])

    # --- Server-side document folder ---
    # Files staged here are indexed by POST /v1/ingest/folder without being uploaded.
    documents_dir: str = os.getenv("DOCUMENTS_DIR", "./data/documents")

    def validate(self) -> None:
        """Fail fast if required configuration is missing or unusable."""
        if self.llm_provider not in ("gemini", "ollama"):
            raise ConfigurationError(
                f"Unknown LLM_PROVIDER '{self.llm_provider}'. Use 'gemini' or 'ollama'."
            )
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not set in .env. Please set GEMINI_API_KEY."
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
