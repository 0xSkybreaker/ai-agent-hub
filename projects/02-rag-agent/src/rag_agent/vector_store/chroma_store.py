"""ChromaDB vector store wrapper for document storage and retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_chroma import Chroma

from rag_agent.config import settings
from rag_agent.utils.logger import get_logger

if TYPE_CHECKING:
    from rag_agent.embeddings.nvidia_embeddings import EmbeddingClient

logger = get_logger()


class ChromaVectorStore:
    """Manages ChromaDB vector storage for document chunks.

    Provides add, search, delete, and statistics operations.
    """

    def __init__(self, embedding_client: EmbeddingClient) -> None:
        """Initialize ChromaDB with persistent storage.

        Args:
            embedding_client: The embedding client used for queries.
        """
        self._embedding_client = embedding_client

        # Ensure persist directory exists
        persist_path = settings.chroma_persist_path
        persist_path.mkdir(parents=True, exist_ok=True)

        self._store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embedding_client._model,
            persist_directory=str(persist_path),
            collection_metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB initialized: collection='{settings.collection_name}' "
            f"path='{persist_path}'"
        )

    def add_documents(self, texts: list[str], metadatas: list[dict]) -> list[str]:
        """Add documents to the vector store.

        ChromaDB handles embedding internally via the embedding function,
        so we pass raw texts rather than pre-computed embeddings.

        Args:
            texts: List of chunk text content.
            metadatas: List of metadata dicts (same length as texts).

        Returns:
            List of document IDs.
        """
        if not texts:
            return []

        logger.info(f"Adding {len(texts)} documents to ChromaDB")
        ids = self._store.add_texts(texts=texts, metadatas=metadatas)
        logger.debug(f"Added {len(ids)} documents, IDs: {ids[:3]}...")
        return ids

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> list[tuple[str, dict, float]]:
        """Semantic similarity search.

        Args:
            query: The query text.
            top_k: Number of results to return (default: settings.top_k).
            filter_metadata: Optional metadata filter for server-side filtering.

        Returns:
            List of (document_text, metadata, relevance_score) tuples.
        """
        k = top_k or settings.top_k
        logger.debug(f"Searching for top-{k} matches: {query[:80]}...")

        results = self._store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=filter_metadata,
        )

        formatted = []
        for doc, score in results:
            formatted.append((doc.page_content, doc.metadata, score))

        logger.debug(f"Found {len(formatted)} results, top score: {formatted[0][2]:.4f}" if formatted else "No results")
        return formatted

    def delete_by_source(self, source_path: str) -> int:
        """Delete all chunks belonging to a specific source file.

        Args:
            source_path: The source file path to match.

        Returns:
            Number of documents deleted.
        """
        # Get all documents matching this source
        collection = self._store._collection
        results = collection.get(where={"source_path": source_path})

        if results and results["ids"]:
            count = len(results["ids"])
            collection.delete(ids=results["ids"])
            logger.info(f"Deleted {count} chunks for source: {source_path}")
            return count

        logger.debug(f"No chunks found for source: {source_path}")
        return 0

    def get_source_hashes(self) -> dict[str, str]:
        """Get content hash for all indexed sources.

        Returns:
            Dict mapping source_path -> content_hash.
        """
        collection = self._store._collection
        results = collection.get()

        hashes: dict[str, str] = {}
        if results and results["metadatas"]:
            for meta in results["metadatas"]:
                source = meta.get("source_path", "")
                chunk_hash = meta.get("content_hash", "")
                if source and source not in hashes:
                    hashes[source] = chunk_hash

        return hashes

    def get_collection_stats(self) -> dict:
        """Return collection statistics.

        Returns:
            Dict with chunk_count and unique_sources.
        """
        collection = self._store._collection
        count = collection.count()

        results = collection.get()
        sources: set[str] = set()
        if results and results["metadatas"]:
            for meta in results["metadatas"]:
                src = meta.get("source_path", "")
                if src:
                    sources.add(src)

        return {
            "chunk_count": count,
            "unique_sources": len(sources),
            "collection_name": settings.collection_name,
        }

    def clear_collection(self) -> None:
        """Delete all documents from the collection."""
        collection = self._store._collection
        count = collection.count()
        if count > 0:
            results = collection.get()
            if results and results["ids"]:
                collection.delete(ids=results["ids"])
            logger.info(f"Cleared collection: {count} documents deleted")
