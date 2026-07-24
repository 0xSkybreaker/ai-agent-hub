"""Configuration via pydantic-settings, loaded from .env and env vars."""

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

    # ── LLM ────────────────────────────────────────────
    llm_base_url: str = Field(default="https://integrate.api.nvidia.com/v1")
    llm_api_key: str = Field(default="...", min_length=1)
    llm_model: str = Field(default="nvidia/llama-3.3-nemotron-super-49b-v1.5")

    # ── Agent ──────────────────────────────────────────
    max_steps: int = Field(default=10, ge=1, le=50)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)


# Singleton — import this everywhere
settings = Settings()
