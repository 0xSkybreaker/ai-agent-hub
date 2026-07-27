"""FastAPI application factory for agentic-rag."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_rag.api.routes import router
from agentic_rag.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # Startup
    print(f"Agentic RAG API starting on {settings.host}:{settings.port}")
    print(f"Model: {settings.chat_model}")
    print(f"ChromaDB: {settings.chroma_persist_dir}")
    yield
    # Shutdown
    print("Agentic RAG API shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Agentic RAG API",
        description=(
            "Agentic RAG — a ReAct Agent that autonomously drives "
            "document retrieval, evaluation, and synthesis."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app


# Module-level app for uvicorn
app = create_app()


def main():
    """Entry point for rag-serve console script."""
    import uvicorn

    uvicorn.run(
        "agentic_rag.api.server:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
