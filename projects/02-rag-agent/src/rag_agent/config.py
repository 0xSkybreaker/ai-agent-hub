"""Application configuration via pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from .env and environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── NVIDIA NIM API ────────────────────────────────────────────
    nvidia_api_key: str = Field(..., min_length=1)
    nvidia_base_url: str = Field(default="https://integrate.api.nvidia.com/v1")
    chat_model: str = Field(default="nvidia/llama-3.3-nemotron-super-49b-v1.5")
    embedding_model: str = Field(default="nvidia/nv-embedqa-e5-v5")

    # ── Vector Store ──────────────────────────────────────────────
    chroma_persist_dir: str = Field(default="./data/chroma")
    collection_name: str = Field(default="documents")

    # ── Chunking ──────────────────────────────────────────────────
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)

    # ── Retrieval ─────────────────────────────────────────────────
    top_k: int = Field(default=5, ge=1, le=100)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    # ── Generation ────────────────────────────────────────────────
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # ── API Server ────────────────────────────────────────────────
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=1, le=65535)

    # ── Memory ────────────────────────────────────────────────────
    max_conversation_turns: int = Field(default=10, ge=1, le=100)

    # ── Logging ───────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    # ── Document Tracker ──────────────────────────────────────────
    tracker_db_path: str = Field(default="./data/tracker.db")

    # ── Derived paths ─────────────────────────────────────────────
    @property
    def chroma_persist_path(self) -> Path:
        return Path(self.chroma_persist_dir).resolve()

    @property
    def tracker_db_path_resolved(self) -> Path:
        return Path(self.tracker_db_path).resolve()


# Singleton instance — import this throughout the application
settings = Settings()
