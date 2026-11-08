from functools import lru_cache
from pathlib import Path
from typing import Optional, Literal
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor the env file to the repository root so starting the service from a
# different working directory cannot silently drop provider credentials.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    # App & API Version
    app_version: str = "1.0.0"

    # Environment & Logging
    env: str = "development"
    log_level: str = "INFO"
    secret_key: str = "weirwise-default-development-secret-key-32chars"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    # Browser origins allowed to call Core directly (JSON list). Empty disables CORS;
    # server-to-server callers such as the Platform are unaffected.
    cors_allow_origins: list[str] = []

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/weirwise"
    database_sync_url: str = "postgresql://postgres:postgres@127.0.0.1:5432/weirwise"
    db_pool_size: int = 30
    db_max_overflow: int = 50
    db_timeout_seconds: int = 60
    db_pool_timeout_seconds: int = 60
    db_pool_recycle_seconds: int = 1800
    db_statement_timeout_ms: int = 30000
    # Connection reuse is required in production; test suites give each test its
    # own event loop, so asyncpg connections must not outlive a single test.
    db_pool_enabled: bool = True

    # Storage
    storage_dir: str = "./var/data/storage"
    max_upload_size_bytes: int = 50 * 1024 * 1024  # 50MB
    upload_read_chunk_bytes: int = 1024 * 1024

    # Ingestion & Chunking
    ingestion_mode: Literal["inline", "worker"] = (
        "inline"  # "inline" (FastAPI background tasks) or "worker" (external worker)
    )
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Background Worker
    worker_poll_interval: float = 2.0
    worker_concurrency: int = 2
    worker_max_retries: int = 3
    worker_lease_seconds: int = 120
    worker_heartbeat_interval: float = 30.0

    # Embedding Provider (OpenAI-compatible)
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_timeout_seconds: float = 30.0
    embedding_batch_size: int = 64
    # Bounded LRU cache for query embeddings (identical text -> vector). Set 0 to disable.
    embedding_cache_size: int = 512
    embedding_cache_ttl_seconds: int = 300
    embedding_pool_connections: int = 20

    # Retrieval
    retrieval_hnsw_ef_search: int = 100

    # Chat / LLM Provider (OpenAI-compatible)
    chat_base_url: str = "https://api.openai.com/v1"
    chat_api_key: Optional[str] = None
    chat_model: str = "gpt-4o-mini"
    chat_timeout_seconds: float = 60.0
    chat_max_tokens: int = 1024
    chat_temperature: float = 0.2

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chunk_size must be a positive integer greater than 0")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int) -> int:
        if v < 0:
            raise ValueError("chunk_overlap cannot be negative")
        return v

    @field_validator("db_pool_size")
    @classmethod
    def validate_pool_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("db_pool_size must be greater than 0")
        return v

    @model_validator(mode="after")
    def validate_chunk_boundaries(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )
        return self

    model_config = SettingsConfigDict(
        env_prefix="WEIRWISE_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
