"""FastAPI route definitions."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from rag_agent import __version__
from rag_agent.api.dependencies import (
    get_generator,
    get_indexer,
    get_memory,
)
from rag_agent.api.models import (
    HealthResponse,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    StatsResponse,
)
from rag_agent.config import settings
from rag_agent.generation.citations import format_citations_for_api
from rag_agent.utils.logger import get_logger

logger = get_logger()
router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check with model info."""
    return HealthResponse(
        status="ok",
        version=__version__,
        model=settings.chat_model,
        embedding_model=settings.embedding_model,
    )


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    """Get indexing and session statistics."""
    indexer = get_indexer()
    memory = get_memory()
    stats = indexer.get_stats()
    return StatsResponse(
        chunk_count=stats["chunk_count"],
        unique_sources=stats["unique_sources"],
        collection_name=stats["collection_name"],
        active_sessions=memory.session_count(),
    )


@router.post("/ingest/file", response_model=IndexResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Upload and index a document file."""
    indexer = get_indexer()

    # Save uploaded file to temp location
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / (file.filename or "uploaded_file")

    content = await file.read()
    file_path.write_bytes(content)

    try:
        result = indexer.index_file(str(file_path))
        return IndexResponse(
            source=result.source,
            chunks_created=result.chunks_created,
            status=result.status,
            message=result.message,
            error=result.error,
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/path", response_model=IndexResponse)
def ingest_path(source_path: str, force: bool = False):
    """Index a file from a local path."""
    indexer = get_indexer()
    result = indexer.index_file(source_path, force=force)

    if result.status == "error":
        raise HTTPException(status_code=400, detail=result.error)

    return IndexResponse(
        source=result.source,
        chunks_created=result.chunks_created,
        status=result.status,
        message=result.message,
        error=result.error,
    )


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Submit a question and get an answer with sources."""
    generator = get_generator()
    memory = get_memory()

    # Create or get session
    session_id = request.session_id or memory.create_session()

    # Get conversation history
    history = memory.get_history(session_id)

    try:
        result = generator.generate(
            question=request.question,
            history=history,
            top_k=request.top_k,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Store in memory
    memory.add_exchange(session_id, request.question, result.answer)

    # Format sources
    sources = [
        SourceDocument(
            file_name=s.file_name,
            source_path=s.source_path,
            page_number=s.page_number,
            chunk_index=s.chunk_index,
            excerpt=s.excerpt,
            relevance_score=round(s.relevance_score, 4),
        )
        for s in result.sources
    ]

    return QueryResponse(
        answer=result.answer,
        sources=sources,
        retrieved_count=result.retrieved_count,
        model=result.model,
        session_id=session_id,
    )


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Submit a question and get a streaming SSE response."""
    generator = get_generator()
    memory = get_memory()

    session_id = request.session_id or memory.create_session()
    history = memory.get_history(session_id)

    # Pre-retrieve for citations
    from rag_agent.api.dependencies import get_retriever
    retriever = get_retriever()
    documents = retriever.retrieve(
        query=request.question,
        top_k=request.top_k,
    )
    citations = format_citations_for_api(
        __import__("rag_agent.generation.citations", fromlist=["extract_citations"]).extract_citations(documents)
    )

    async def event_generator():
        full_answer: list[str] = []

        try:
            for token in generator.generate_stream(
                question=request.question,
                history=history,
                top_k=request.top_k,
            ):
                full_answer.append(token)
                yield {"event": "token", "data": json.dumps({"token": token})}

            # Store in memory
            answer_text = "".join(full_answer)
            memory.add_exchange(session_id, request.question, answer_text)

            # Send final event with citations
            yield {
                "event": "done",
                "data": json.dumps({
                    "sources": citations,
                    "session_id": session_id,
                }),
            }

        except Exception as e:
            logger.error(f"Stream query failed: {e}")
            yield {"event": "error", "data": json.dumps({"detail": str(e)})}

    return EventSourceResponse(event_generator())


@router.delete("/memory/{session_id}")
def clear_memory(session_id: str):
    """Clear conversation history for a session."""
    memory = get_memory()
    memory.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.delete("/documents/{source_path:path}")
def delete_document(source_path: str):
    """Remove a document from the index."""
    indexer = get_indexer()
    count = indexer.remove_document(source_path)
    return {"status": "deleted", "chunks_removed": count, "source": source_path}
