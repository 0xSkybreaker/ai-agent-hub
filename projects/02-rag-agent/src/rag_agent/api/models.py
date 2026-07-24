"""Pydantic models for API requests and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request for document Q&A."""

    question: str = Field(..., min_length=1, max_length=5000, description="The question to answer")
    session_id: str | None = Field(None, description="Session ID for conversation history")
    top_k: int | None = Field(None, ge=1, le=100, description="Number of documents to retrieve")
    stream: bool = Field(False, description="Enable SSE streaming response")


class SourceDocument(BaseModel):
    """A retrieved source document."""

    file_name: str
    source_path: str
    page_number: int | None = None
    chunk_index: int | None = None
    excerpt: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Response for a Q&A query."""

    answer: str
    sources: list[SourceDocument] = []
    retrieved_count: int = 0
    model: str = ""
    session_id: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = ""
    model: str = ""
    embedding_model: str = ""


class StatsResponse(BaseModel):
    """Index statistics response."""

    chunk_count: int = 0
    unique_sources: int = 0
    collection_name: str = ""
    active_sessions: int = 0


class IndexResponse(BaseModel):
    """Response for document indexing."""

    source: str
    chunks_created: int = 0
    status: str  # "indexed", "updated", "unchanged", "error"
    message: str = ""
    error: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
