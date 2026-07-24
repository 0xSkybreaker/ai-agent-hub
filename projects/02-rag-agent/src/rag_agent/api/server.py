"""FastAPI application factory and server entry point."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_agent import __version__
from rag_agent.api.routes import router
from rag_agent.config import settings
from rag_agent.utils.logger import get_logger

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize and cleanup resources."""
    logger.info(f"Starting RAG Agent v{__version__}")
    logger.info(f"Model: {settings.chat_model}")
    logger.info(f"Embedding: {settings.embedding_model}")

    # Ensure data directories exist
    Path("data/chroma").mkdir(parents=True, exist_ok=True)
    Path("data/uploads").mkdir(parents=True, exist_ok=True)

    yield

    logger.info("Shutting down RAG Agent")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app instance.
    """
    app = FastAPI(
        title="RAG Agent",
        description="Document Q&A using NVIDIA NIM and ChromaDB",
        version=__version__,
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

    # Routes
    app.include_router(router)

    return app


# Module-level app instance for uvicorn
app = create_app()


def main():
    """Entry point for `rag-serve` console script."""
    import uvicorn

    logger.info(f"Starting API server on {settings.host}:{settings.port}")
    uvicorn.run(
        "rag_agent.api.server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
