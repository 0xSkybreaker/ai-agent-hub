"""FastAPI route definitions for agentic-rag."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from agentic_rag import __version__
from agentic_rag.api.dependencies import get_agent
from agentic_rag.api.models import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
    StepInfo,
)
from agentic_rag.config import settings

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check with model and config info."""
    return HealthResponse(
        status="ok",
        version=__version__,
        model=settings.chat_model,
        max_steps=settings.max_steps,
    )


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Submit a question and get an answer with step trace and sources.

    The agent will autonomously search, evaluate, and synthesize
    before returning the final answer.
    """
    agent = get_agent()

    try:
        result = agent.run(
            task=request.question,
            max_steps=request.max_steps,
            verbose=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Convert steps to API-friendly format
    steps = [
        StepInfo(
            step_number=s.step_number,
            tool_name=s.tool_name if s.tool_name else None,
            tool_args=s.tool_args if s.tool_args else None,
            tool_result_preview=s.tool_result[:300] if s.tool_result else None,
            final_answer=s.final_answer if s.final_answer else None,
            is_final=s.is_final,
        )
        for s in result.steps
    ]

    sources = [
        SourceInfo(file_name=s["file_name"], source_path=s["source_path"])
        for s in result.sources
    ]

    return QueryResponse(
        answer=result.answer,
        steps=steps,
        total_steps=result.total_steps,
        sources=sources,
        model=result.model,
    )


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Submit a question and get a streaming SSE response.

    Events:
        step  — {"step_number": N, "tool_name": "...", "tool_args": {...}, "tool_result": "..."}
        done  — {"answer": "...", "sources": [...], "total_steps": N, "model": "..."}
        error — {"detail": "..."}
    """
    agent = get_agent()

    async def event_generator():
        try:
            for event in agent.run_stream(
                task=request.question,
                max_steps=request.max_steps,
            ):
                if event["type"] == "step":
                    s = event["step"]
                    yield {
                        "event": "step",
                        "data": json.dumps({
                            "step_number": s.step_number,
                            "tool_name": s.tool_name if s.tool_name else None,
                            "tool_args": s.tool_args if s.tool_args else None,
                            "tool_result": s.tool_result[:500] if s.tool_result else None,
                            "final_answer": s.final_answer if s.final_answer else None,
                            "is_final": s.is_final,
                        }),
                    }
                elif event["type"] == "done":
                    r = event["result"]
                    yield {
                        "event": "done",
                        "data": json.dumps({
                            "answer": r.answer,
                            "sources": r.sources,
                            "total_steps": r.total_steps,
                            "model": r.model,
                        }),
                    }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"detail": str(e)}),
            }

    return EventSourceResponse(event_generator())
