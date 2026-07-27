"""Pydantic models for API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request for agentic RAG Q&A."""

    question: str = Field(..., min_length=1, max_length=5000, description="The question to answer")
    max_steps: int | None = Field(None, ge=1, le=50, description="Maximum agent iterations")
    stream: bool = Field(False, description="Enable SSE streaming response")


class SourceInfo(BaseModel):
    """Information about a source document used in the answer."""

    file_name: str
    source_path: str


class StepInfo(BaseModel):
    """A single step in the agent's trace."""

    step_number: int
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result_preview: str | None = None  # Truncated for API
    final_answer: str | None = None
    is_final: bool = False


class QueryResponse(BaseModel):
    """Response for an agentic RAG query."""

    answer: str
    steps: list[StepInfo] = []
    total_steps: int = 0
    sources: list[SourceInfo] = []
    model: str = ""


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = ""
    model: str = ""
    max_steps: int = 0


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
