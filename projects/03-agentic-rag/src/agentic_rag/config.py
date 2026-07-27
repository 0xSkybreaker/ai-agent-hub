"""Configuration for agentic-rag via pydantic-settings.

Separate from rag-agent config — this project has its own settings
with agent-specific fields (max_steps, etc.) but shares the same
NVIDIA NIM API credentials and ChromaDB data directory.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config — reads from .env and environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── NVIDIA NIM API ────────────────────────────────────────────
    nvidia_api_key: str = Field(default="...", min_length=1)
    nvidia_base_url: str = Field(default="https://integrate.api.nvidia.com/v1")
    chat_model: str = Field(default="nvidia/llama-3.3-nemotron-super-49b-v1.5")

    # ── Agent ─────────────────────────────────────────────────────
    max_steps: int = Field(default=15, ge=1, le=50)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)

    # ── Retrieval ─────────────────────────────────────────────────
    top_k: int = Field(default=5, ge=1, le=100)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    # ── Vector Store (shared with rag-agent) ──────────────────────
    chroma_persist_dir: str = Field(default="../02-rag-agent/data/chroma")
    collection_name: str = Field(default="documents")

    # ── API Server ────────────────────────────────────────────────
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8001, ge=1, le=65535)


# Singleton — import this everywhere
settings = Settings()
