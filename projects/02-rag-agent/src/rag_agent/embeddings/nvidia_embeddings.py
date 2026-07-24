"""NVIDIA NIM embedding client wrapper."""

from __future__ import annotations

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from rag_agent.config import settings
from rag_agent.utils.logger import get_logger

logger = get_logger()


class EmbeddingClient:
    """Wraps NVIDIA embedding model with batching support.

    Uses LangChain's NVIDIAEmbeddings which handles API auth, retries,
    and the OpenAI-compatible format natively.
    """

    def __init__(self) -> None:
        self._model = NVIDIAEmbeddings(
            model=settings.embedding_model,
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
        )
        self.model_name = settings.embedding_model
        logger.info(f"Embedding client initialized: model={self.model_name}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of document texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        if not texts:
            return []

        logger.debug(f"Embedding {len(texts)} documents")
        embeddings = self._model.embed_documents(texts)
        logger.debug(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query text.

        Args:
            text: The query text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        logger.debug(f"Embedding query: {text[:80]}...")
        return self._model.embed_query(text)
